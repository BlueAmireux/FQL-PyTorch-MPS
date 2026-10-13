"""Bounded MPS supervised transfer from the frozen state FQL action policy."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from fql_torch.runtime import configure_device, memory, synchronize, runtime_record
from visual.common import (load_teacher, load_student_checkpoint, save_student_checkpoint,
                           sha256, write_json, visual_hashes)
from visual.student import VisualStudent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True, help='Total completed updates, including a resumed prefix.')
    parser.add_argument('--resume', type=int, default=0)
    args = parser.parse_args()
    device = configure_device('mps', 4)
    teacher, identity = load_teacher('mps')
    def teacher_digest():
        digest = hashlib.sha256()
        for name, value in teacher.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
    initial_teacher_digest = teacher_digest()
    manifest = json.loads((args.data / 'manifest.json').read_text())
    arrays = {}
    for split in ('train', 'val'):
        path = args.data / f'{split}.npz'
        if sha256(path) != manifest['splits'][split]['sha256']:
            raise ValueError('Dataset hash mismatch.')
        with np.load(path) as data:
            arrays[split] = {k: torch.as_tensor(data[k], device=device) for k in ('images', 'proprio', 'states')}
    config = {'seed': 613, 'batch_size': 256, 'learning_rate': 1e-4, 'optimizer': 'Adam fused',
              'data': str(args.data.resolve()), 'manifest_sha256': sha256(args.data / 'manifest.json'),
              'loss': 'MSE(unclipped student, frozen unclipped teacher), same fresh Gaussian z',
              'validation_noise_seed': 614, 'image_shuffle_seed': 615,
              'supervised_distillation': True, 'FQL_RL_updates_added': 0}
    if args.resume:
        model, payload = load_student_checkpoint(args.run, args.resume, device, resume=False)
        if config != payload['config'] or identity['sha256'] != payload['teacher']['sha256']:
            raise ValueError('Resume config or teacher mismatch.')
        if payload['device'] != device.type:
            raise ValueError('Training resume requires the same device.')
        sampler = np.random.default_rng()
        sampler.bit_generator.state = payload['sampler']
    else:
        args.run.mkdir(parents=True, exist_ok=False)
        # Teacher's checkpoint constructor resets RNG; set the new seed afterwards.
        torch.manual_seed(config['seed'])
        model = VisualStudent(teacher.state_dict()).to(device)
        sampler = np.random.default_rng(config['seed'])
        write_json(args.run / 'config.json', config, exclusive=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], fused=True)
    if args.resume:
        optimizer.load_state_dict(payload['optimizer'])
        from fql_torch.checkpoint import restore_rng
        restore_rng(payload['rng'], device)
    if args.steps <= args.resume:
        raise ValueError('Target must exceed resume step.')
    validation_generator = torch.Generator(device='cpu').manual_seed(614)
    validation_noises = torch.randn((len(arrays['val']['images']), 5), generator=validation_generator).to(device)
    permutation = torch.randperm(len(validation_noises), generator=torch.Generator().manual_seed(615)).to(device)

    @torch.inference_mode()
    def validate(image_mode='normal'):
        model.eval()
        raw_errors, clipped_errors, teacher_saturation, student_saturation = [], [], [], []
        val = arrays['val']
        for start in range(0, len(validation_noises), 256):
            stop = min(start + 256, len(validation_noises))
            images = val['images'][start:stop]
            if image_mode == 'shuffle':
                images = val['images'][permutation[start:stop]]
            elif image_mode == 'zero':
                images = torch.zeros_like(images)
            predicted = model.raw_actions(images, val['proprio'][start:stop], validation_noises[start:stop])
            target = teacher(val['states'][start:stop], validation_noises[start:stop])
            raw_errors.append((predicted - target).square().cpu())
            clipped_errors.append((predicted.clamp(-1, 1) - target.clamp(-1, 1)).square().cpu())
            teacher_saturation.append((target.abs() >= 1).cpu())
            student_saturation.append((predicted.abs() >= 1).cpu())
        raw, clipped = torch.cat(raw_errors), torch.cat(clipped_errors)
        model.train()
        return {'raw_mse': raw.mean().item(), 'clipped_mse': clipped.mean().item(),
                'raw_mse_by_action': raw.mean(0).tolist(),
                'clipped_mse_by_action': clipped.mean(0).tolist(), 'image_mode': image_mode,
                'teacher_saturation_by_action': torch.cat(teacher_saturation).float().mean(0).tolist(),
                'student_saturation_by_action': torch.cat(student_saturation).float().mean(0).tolist(),
                'samples': len(validation_noises), 'fixed_noise': True}

    segment = {'start_step': args.resume, 'target_step': args.steps, 'teacher': identity,
               'source_sha256': visual_hashes(), 'runtime': runtime_record(device),
               'initial_validation': validate(), 'first_update': {}, 'validations': []}
    train = arrays['train']
    model.train()
    started = last_record = time.perf_counter()
    last_step = args.resume
    with (args.run / f'train_{args.resume}_{args.steps}.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['step', 'loss', 'updates_per_second', 'elapsed_seconds'])
        writer.writeheader()
        for step in range(args.resume + 1, args.steps + 1):
            indices = torch.as_tensor(sampler.integers(0, len(train['images']), 256), device=device)
            images, proprio, states = (train[key][indices] for key in ('images', 'proprio', 'states'))
            noises = torch.randn((256, 5), device=device)
            with torch.no_grad():
                targets = teacher(states, noises)
            optimizer.zero_grad(set_to_none=True)
            predictions = model.raw_actions(images, proprio, noises)
            loss = (predictions - targets).square().mean()
            loss.backward()
            if step == args.resume + 1:
                encoder_before = model.encoder[0].weight.detach().clone()
                segment['first_update'] = {'loss_before': loss.item(),
                    'all_gradients_finite': all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()),
                    'encoder_gradient_norm': torch.linalg.vector_norm(model.encoder[0].weight.grad).item(),
                    'teacher_gradients_absent': all(p.grad is None and not p.requires_grad for p in teacher.parameters())}
                if not segment['first_update']['all_gradients_finite']:
                    raise ValueError('Nonfinite first real update.')
            optimizer.step()
            if step == args.resume + 1:
                with torch.no_grad():
                    after = (model.raw_actions(images, proprio, noises) - targets).square().mean().item()
                segment['first_update']['same_batch_loss_after'] = after
                segment['first_update']['encoder_max_abs_parameter_change'] = (model.encoder[0].weight - encoder_before).abs().max().item()
            if step % 250 == 0 or step == args.steps:
                synchronize(device)
                now = time.perf_counter()
                row = {'step': step, 'loss': loss.item(), 'updates_per_second': (step - last_step) / (now - last_record),
                       'elapsed_seconds': now - started}
                if not np.isfinite(row['loss']):
                    raise ValueError('Nonfinite training loss.')
                writer.writerow(row)
                stream.flush()
                print(row, flush=True)
                last_record, last_step = now, step
            if step in (1000, 5000, 10000, 20000):
                segment['validations'].append({'step': step, **validate()})
        synchronize(device)
        training_seconds = time.perf_counter() - started
    probe = arrays['val']
    save_student_checkpoint(args.run, args.steps, model, optimizer, config, sampler, identity,
                            probe['images'][:4], probe['proprio'][:4], validation_noises[:4])
    segment.update(completed_step=args.steps, training_seconds=training_seconds,
                   mean_updates_per_second=(args.steps - args.resume) / training_seconds,
                   memory=memory(device), final_validation={mode: validate(mode) for mode in ('normal', 'shuffle', 'zero')},
                   teacher_parameters_unchanged=teacher_digest() == initial_teacher_digest,
                   teacher_initial_parameter_sha256=initial_teacher_digest, environment_steps_added=0)
    if not segment['teacher_parameters_unchanged']:
        raise ValueError('Frozen teacher parameters changed.')
    write_json(args.run / f'segment_{args.resume}_{args.steps}.json', segment, exclusive=True)
    print(json.dumps({key: segment[key] for key in ('completed_step', 'training_seconds', 'mean_updates_per_second', 'first_update', 'final_validation', 'memory')}), flush=True)


if __name__ == '__main__':
    main()
