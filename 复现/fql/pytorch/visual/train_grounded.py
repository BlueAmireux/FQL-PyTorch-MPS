"""Controlled action distillation plus image-only cube XYZ supervision.

Only encoder outputs 0:3 receive geometry labels; quaternion/yaw remain latent.
Inference uses the unchanged VisualStudent image/proprio/noise interface.
"""
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
from visual.common import (load_teacher, save_student_checkpoint, sha256,
                           write_json, visual_hashes)
from visual.student import VisualStudent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True, help='New updates; no resume in this branch.')
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError('steps must be positive.')
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
            arrays[split] = {k: torch.as_tensor(data[k], device=device)
                             for k in ('images', 'proprio', 'states')}
    config = {'seed': 613, 'batch_size': 256, 'learning_rate': 1e-4, 'optimizer': 'Adam fused',
              'data': str(args.data.resolve()), 'manifest_sha256': sha256(args.data / 'manifest.json'),
              'loss': 'raw same-z action MSE + 1.0 * image-only XYZ MSE',
              'xyz_loss_weight': 1.0, 'xyz_target': 'unaltered states[:,19:22], centered xyz * 10',
              'xyz_rmse_cm_scale': 10.0, 'validation_noise_seed': 614, 'image_shuffle_seed': 615,
              'supervised_distillation': True, 'FQL_RL_updates_added': 0,
              'additional_source_sha256': {'train_grounded.py': sha256(Path(__file__))},
              'checkpoint_loader_scope': 'common checks student/common/train; independent acceptance must also check additional_source_sha256'}
    args.run.mkdir(parents=True, exist_ok=False)
    # Match the baseline initialization after load_teacher resets Torch RNG.
    torch.manual_seed(config['seed'])
    model = VisualStudent(teacher.state_dict()).to(device)
    sampler = np.random.default_rng(config['seed'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], fused=True)
    write_json(args.run / 'config.json', config, exclusive=True)
    validation_noises = torch.randn((len(arrays['val']['images']), 5),
                                   generator=torch.Generator().manual_seed(614)).to(device)
    permutation = torch.randperm(len(validation_noises),
                                 generator=torch.Generator().manual_seed(615)).to(device)

    @torch.inference_mode()
    def validate(image_mode='normal'):
        model.eval()
        raw_errors, clipped_errors, xyz_errors = [], [], []
        teacher_saturation, student_saturation = [], []
        val = arrays['val']
        for start in range(0, len(validation_noises), 256):
            stop = min(start + 256, len(validation_noises))
            images = val['images'][start:stop]
            if image_mode == 'shuffle':
                images = val['images'][permutation[start:stop]]
            elif image_mode == 'zero':
                images = torch.zeros_like(images)
            visual = model.encoder(images.float() / 255.0)
            predicted = model.actor(torch.cat((val['proprio'][start:stop], visual), dim=-1),
                                    validation_noises[start:stop])
            target = teacher(val['states'][start:stop], validation_noises[start:stop])
            raw_errors.append((predicted - target).square().cpu())
            clipped_errors.append((predicted.clamp(-1, 1) - target.clamp(-1, 1)).square().cpu())
            xyz_errors.append((visual[:, :3] - val['states'][start:stop, 19:22]).square().cpu())
            teacher_saturation.append((target.abs() >= 1).cpu())
            student_saturation.append((predicted.abs() >= 1).cpu())
        raw, clipped, xyz = map(torch.cat, (raw_errors, clipped_errors, xyz_errors))
        if not all(torch.isfinite(values).all().item() for values in (raw, clipped, xyz)):
            raise ValueError('Nonfinite validation error.')
        model.train()
        return {'raw_mse': raw.mean().item(), 'clipped_mse': clipped.mean().item(),
                'raw_mse_by_action': raw.mean(0).tolist(), 'clipped_mse_by_action': clipped.mean(0).tolist(),
                'xyz_axis_rmse_cm': (xyz.mean(0).sqrt() * 10).tolist(),
                'xyz_euclidean_rmse_cm': (xyz.sum(1).mean().sqrt() * 10).item(),
                'teacher_saturation_by_action': torch.cat(teacher_saturation).float().mean(0).tolist(),
                'student_saturation_by_action': torch.cat(student_saturation).float().mean(0).tolist(),
                'image_mode': image_mode, 'samples': len(validation_noises), 'fixed_noise': True}

    segment = {'start_step': 0, 'target_step': args.steps, 'teacher': identity,
               'source_sha256': visual_hashes(), 'additional_source_sha256': config['additional_source_sha256'],
               'runtime': runtime_record(device), 'initial_validation': validate(),
               'first_update': {}, 'validations': []}
    print({'step': 0, 'validation': segment['initial_validation']}, flush=True)
    train = arrays['train']
    model.train()
    started = last_record = time.perf_counter()
    last_step = 0
    with (args.run / f'train_0_{args.steps}.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['step', 'loss', 'action_mse', 'xyz_mse',
                                                   'updates_per_second', 'elapsed_seconds'])
        writer.writeheader()
        for step in range(1, args.steps + 1):
            indices = torch.as_tensor(sampler.integers(0, len(train['images']), 256), device=device)
            images, proprio, states = (train[key][indices] for key in ('images', 'proprio', 'states'))
            noises = torch.randn((256, 5), device=device)
            with torch.no_grad():
                targets = teacher(states, noises)
            optimizer.zero_grad(set_to_none=True)
            visual = model.encoder(images.float() / 255.0)
            predictions = model.actor(torch.cat((proprio, visual), dim=-1), noises)
            action_mse = (predictions - targets).square().mean()
            xyz_mse = (visual[:, :3] - states[:, 19:22]).square().mean()
            loss = action_mse + xyz_mse
            loss.backward()
            if step == 1:
                encoder_before = model.encoder[0].weight.detach().clone()
                segment['first_update'] = {'loss_before': loss.item(), 'action_mse': action_mse.item(),
                    'xyz_mse': xyz_mse.item(),
                    'all_gradients_finite': all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()),
                    'encoder_gradient_norm': model.encoder[0].weight.grad.norm().item(),
                    'teacher_gradients_absent': all(p.grad is None and not p.requires_grad for p in teacher.parameters())}
                if not segment['first_update']['all_gradients_finite']:
                    raise ValueError('Nonfinite first grounded update.')
            optimizer.step()
            if step == 1:
                segment['first_update']['encoder_max_abs_parameter_change'] = (model.encoder[0].weight - encoder_before).abs().max().item()
            if (step <= 1000 and step % 250 == 0) or step % 1000 == 0 or step == args.steps:
                synchronize(device)
                now = time.perf_counter()
                row = {'step': step, 'loss': loss.item(), 'action_mse': action_mse.item(), 'xyz_mse': xyz_mse.item(),
                       'updates_per_second': (step - last_step) / (now - last_record), 'elapsed_seconds': now - started}
                if not np.isfinite([row['loss'], row['action_mse'], row['xyz_mse']]).all():
                    raise ValueError('Nonfinite grounded training loss.')
                writer.writerow(row)
                stream.flush()
                print(row, flush=True)
                last_record, last_step = now, step
            if step in (1000, 5000, 10000, 20000):
                result = {'step': step, **validate()}
                segment['validations'].append(result)
                print({'validation': result}, flush=True)
            if step == 1000:
                seconds = time.perf_counter() - started
                segment['first_1000'] = {'training_seconds': seconds, 'updates_per_second': 1000 / seconds,
                                        'includes': 'first-step checks and normal validation; excludes checkpoint write'}
                print({'first_1000': segment['first_1000']}, flush=True)
                if step < args.steps:
                    save_student_checkpoint(args.run, step, model, optimizer, config, sampler, identity,
                                            arrays['val']['images'][:4], arrays['val']['proprio'][:4], validation_noises[:4])
        synchronize(device)
        training_seconds = time.perf_counter() - started
    save_student_checkpoint(args.run, args.steps, model, optimizer, config, sampler, identity,
                            arrays['val']['images'][:4], arrays['val']['proprio'][:4], validation_noises[:4])
    segment.update(completed_step=args.steps, training_seconds=training_seconds,
                   mean_updates_per_second=args.steps / training_seconds, memory=memory(device),
                   final_validation={mode: validate(mode) for mode in ('normal', 'shuffle', 'zero')},
                   teacher_parameters_unchanged=teacher_digest() == initial_teacher_digest,
                   teacher_initial_parameter_sha256=initial_teacher_digest, environment_steps_added=0)
    if not segment['teacher_parameters_unchanged']:
        raise ValueError('Frozen teacher parameters changed.')
    write_json(args.run / f'segment_0_{args.steps}.json', segment, exclusive=True)
    print(json.dumps({key: segment[key] for key in ('completed_step', 'training_seconds', 'mean_updates_per_second',
                                                   'first_update', 'final_validation', 'memory')}), flush=True)


if __name__ == '__main__':
    main()
