"""Fixed-count full-student-trajectory replay; same frozen grounded parent.

Copied from train_matched_replay.py. Only the student replay input window and
its required checks differ; losses, Adam, Torch RNG and sampling loop stay fixed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from fql_torch.checkpoint import restore_rng
from fql_torch.runtime import configure_device, memory, runtime_record, synchronize
from visual.common import (load_student_checkpoint, load_teacher, save_student_checkpoint,
                           sha256, visual_hashes, write_json)

BASE_SOURCE_SHA = 'ad5b08335629e3c67215afd7f6e2dd9fb04ce6a445fe48fe315bbb9fd4c4048e'
PARENT_SHA = 'a9195b17ac78c6be3a7aa0937c205d2c6345f7d2e786059cfd565b764d6a38f8'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent-run', 'old-data', 'val-data', 'run'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--new-data', type=Path, nargs='+', required=True, help='The fixed-count full-student-window training collection.')
    parser.add_argument('--parent-step', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True, help='Additional updates in this correction stage.')
    parser.add_argument('--arm', choices=('student_full',), required=True)
    args = parser.parse_args()
    if args.parent_step != 20000 or args.steps != 20000 or len(args.new_data) != 1:
        raise ValueError('Require grounded20k, 20k new updates and one matched collection.')
    if sha256(args.parent_run / 'checkpoint_20000.pt') != PARENT_SHA:
        raise ValueError('Ablation parent must be the frozen grounded20k checkpoint.')
    base_source = Path(__file__).parent / 'train_matched_replay.py'
    if sha256(base_source) != BASE_SOURCE_SHA:
        raise ValueError('Copied correction-trainer source identity changed.')
    device = configure_device('mps', 4)
    teacher, identity = load_teacher('mps')
    model, parent = load_student_checkpoint(args.parent_run, args.parent_step, device, resume=False)
    parent_config = parent['config']
    if parent_config != json.loads((args.parent_run / 'config.json').read_text()):
        raise ValueError('Parent run and checkpoint configurations differ.')
    if parent['device'] != device.type or parent['teacher']['sha256'] != identity['sha256']:
        raise ValueError('Parent device or teacher identity mismatch.')
    if parent_config.get('xyz_loss_weight') != 1.0 or parent_config['learning_rate'] != 1e-4:
        raise ValueError('Require the grounded loss and original learning rate.')
    if 'arm' in parent_config:
        raise ValueError('Both ablations must start from the original grounded parent, not a correction arm.')
    extra_sources = dict(parent_config['additional_source_sha256'])
    for name, digest in extra_sources.items():
        if sha256(Path(__file__).parent / name) != digest:
            raise ValueError(f'Additional parent source changed: {name}')
    extra_sources['train_temporal_coverage.py'] = sha256(Path(__file__))
    collector_path = Path(__file__).parent / 'collect_corrections.py'
    extra_sources['collect_corrections.py'] = sha256(collector_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, fused=True)
    optimizer.load_state_dict(parent['optimizer'])
    if any(group['lr'] != 1e-4 or not group['fused'] for group in optimizer.param_groups):
        raise ValueError('Warm-start Adam must retain lr=1e-4 and fused=True.')

    def teacher_digest():
        digest = hashlib.sha256()
        for name, value in teacher.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    initial_teacher_digest = teacher_digest()
    manifests = {name: json.loads((path / 'manifest.json').read_text()) for name, path in
                 (('old', args.old_data), ('val', args.val_data))}
    new_manifests = [json.loads((path / 'manifest.json').read_text()) for path in args.new_data]
    for record, split in [(record, 'train') for record in new_manifests] + [(manifests['val'], 'val')]:
        if record['status'] != 'completed' or record['split'] != split or record['teacher']['sha256'] != identity['sha256']:
            raise ValueError('Collection split, completion, or teacher mismatch.')
        for path, digest in record['source_sha256'].items():
            if sha256(Path(path)) != digest:
                raise ValueError(f'Collection source changed: {path}')
        if record['source_sha256'].get(str(collector_path.resolve())) != extra_sources['collect_corrections.py']:
            raise ValueError('Collector identity differs across declared collections.')

    def load_arrays(path, expected_sha):
        if sha256(path) != expected_sha:
            raise ValueError(f'Data hash mismatch: {path}')
        with np.load(path, allow_pickle=False) as data:
            images, proprio, states = (data[key] for key in ('images', 'proprio', 'states'))
            count = len(images)
            if (count == 0 or images.dtype != np.uint8 or images.shape != (count, 3, 64, 64)
                    or states.dtype != np.float32 or states.shape != (count, 28)
                    or proprio.dtype != np.float32 or proprio.shape != (count, 19)
                    or not np.array_equal(proprio, states[:, :19]) or not np.isfinite(states).all()):
                raise ValueError('Invalid RGB/state/proprio input contract.')
            arrays = {key: torch.as_tensor(data[key], device=device) for key in ('images', 'proprio', 'states')}
            metadata = {key: data[key].copy() for key in ('behavior_id', 'env_seeds', 'episode_ids', 'pair_ids', 'step_index', 'noises') if key in data}
        return arrays, metadata

    old, _ = load_arrays(args.old_data / 'train.npz', manifests['old']['splits']['train']['sha256'])
    old_val, _ = load_arrays(args.old_data / 'val.npz', manifests['old']['splits']['val']['sha256'])
    new_parts = [load_arrays(path / 'data.npz', record['data_sha256']) for path, record in zip(args.new_data, new_manifests)]
    new = {key: torch.cat([part[0][key] for part in new_parts]) for key in ('images', 'proprio', 'states')}
    new_val, val_meta = load_arrays(args.val_data / 'data.npz', manifests['val']['data_sha256'])
    expected_old_manifest = parent_config['manifest_sha256']
    if isinstance(expected_old_manifest, dict):
        expected_old_manifest = expected_old_manifest['old']
    if sha256(args.old_data / 'manifest.json') != expected_old_manifest:
        raise ValueError('Old-data identity must match the parent training source.')
    training_seeds = set(np.concatenate([part[1]['env_seeds'] for part in new_parts]))
    if len(old_val['images']) != 1000 or training_seeds & set(val_meta['env_seeds']):
        raise ValueError('Expected official val1000 and whole-episode collection holdout.')
    for record, metadata in list(zip(new_manifests, [part[1] for part in new_parts])) + [(manifests['val'], val_meta)]:
        if (len(metadata['env_seeds']) != record['completed_episodes']
                or not np.array_equal(np.unique(metadata['episode_ids']), np.arange(len(metadata['env_seeds'])))):
            raise ValueError('Incomplete collection episode coverage.')
    behaviors = np.concatenate([part[1]['behavior_id'] for part in new_parts])
    student_ids, teacher_ids = (np.flatnonzero(behaviors == category) for category in (0, 1))
    if (not np.issubdtype(behaviors.dtype, np.integer) or len(behaviors) != len(new['images'])
            or not np.isin(behaviors, [0, 1]).all() or min(len(student_ids), len(teacher_ids)) == 0):
        raise ValueError('Require both student correction and teacher demonstration samples.')
    record, metadata = new_manifests[0], new_parts[0][1]
    if (record['paired_cases'] != 20 or record['samples_per_behavior'] != len(student_ids)
            or len(student_ids) != len(teacher_ids) or manifests['val']['paired_cases'] != 6
            or len(new_val['images']) != 2 * manifests['val']['samples_per_behavior']):
        raise ValueError('Require 20 train pairs, six heldout pairs and equal nonempty behavior pools.')
    if not np.array_equal(metadata['pair_ids'][student_ids], metadata['pair_ids'][teacher_ids]):
        raise ValueError('Per-case counts/order must stay matched.')
    if (record.get('student_replay_window') != 'whole_raw_trajectory_uniform_subsample'
            or record.get('teacher_pool_exact_copy') is not True):
        raise ValueError('Require full-student window and the frozen teacher prefix pool.')
    if record.get('selection_by_success') is not False or manifests['val'].get('selection_by_success') is not False:
        raise ValueError('Both matched splits must retain all initial cases.')
    pool = student_ids
    category = 'student'
    pools = (pool, pool)
    episode_ids = np.concatenate([part[1]['episode_ids'] for part in new_parts])
    stage, sampling_seed = 1, 617
    old_rng, new_rng = map(np.random.default_rng, np.random.SeedSequence(sampling_seed).spawn(2))
    config = {'arm': args.arm, 'correction_stage': stage, 'sampling_seed': sampling_seed,
              'sampler_schema': 'dual-PCG64: payload.sampler.old/new; SeedSequence(seed).spawn(2)',
              'sampling_change': 'Restore parent Torch RNG; replace parent dataset sampler with new declared streams.',
              'parent_run': str(args.parent_run.resolve()), 'parent_step': args.parent_step,
              'parent_checkpoint_sha256': sha256(args.parent_run / f'checkpoint_{args.parent_step}.pt'),
              'batch_size': 256, 'learning_rate': 1e-4, 'optimizer': 'Adam fused', 'xyz_loss_weight': 1.0,
              'loss': 'raw same-z action MSE + 1.0 * image-only XYZ MSE',
              'mix': f'common old128 + {category}64 + {category}64', 'new_data_category': category,
              'available_unique_sample_pools': {'old': len(old['images']), 'student': len(student_ids), 'teacher': len(teacher_ids)},
              'selected_new_pool': {'category': category, 'behavior_id': 0,
                                    'unique_samples': len(np.unique(pool)),
                                    'episodes': len(np.unique(episode_ids[pool]))},
              'copied_from_source_sha256': BASE_SOURCE_SHA,
              'matching': {'paired_cases': record['paired_cases'], 'frames_per_behavior': len(student_ids),
                           'rule': record['matching_rule'], 'student_window': record['student_replay_window'],
                           'teacher_pool_exact_copy': True, 'selection_by_success': False,
                           'same_new_sampler_indices': True, 'limits': record['limitations']},
              'data': {'old': str(args.old_data.resolve()), 'new': [str(path.resolve()) for path in args.new_data], 'val': str(args.val_data.resolve())},
              'manifest_sha256': {'old': sha256(args.old_data / 'manifest.json'), 'new': [sha256(path / 'manifest.json') for path in args.new_data], 'val': sha256(args.val_data / 'manifest.json')},
              'validation_noise_seeds': {'old': 614, 'new': 1617}, 'additional_source_sha256': extra_sources,
              'checkpoint_loader_scope': 'common checks baseline3; independent audit must check all additional_source_sha256',
              'supervised_distillation': True, 'FQL_RL_updates_added': 0}
    args.run.mkdir(parents=True, exist_ok=False)
    write_json(args.run / 'config.json', config, exclusive=True)
    val_sets = {'old': old_val, 'new': new_val}
    val_noises = {name: torch.randn((len(data['images']), 5), generator=torch.Generator().manual_seed(seed)).to(device)
                  for (name, data), seed in zip(val_sets.items(), (614, 1617))}
    permutations = {name: torch.randperm(len(data['images']), generator=torch.Generator().manual_seed(seed)).to(device)
                    for (name, data), seed in zip(val_sets.items(), (615, 1618))}

    @torch.inference_mode()
    def validate(name, mode='normal'):
        model.eval()
        data, noises = val_sets[name], val_noises[name]
        raw_errors, clip_errors, xyz_errors = [], [], []
        for start in range(0, len(noises), 256):
            stop = min(start + 256, len(noises))
            images = data['images'][permutations[name][start:stop]] if mode == 'shuffle' else data['images'][start:stop]
            if mode == 'zero':
                images = torch.zeros_like(images)
            visual = model.encoder(images.float() / 255)
            predicted = model.actor(torch.cat((data['proprio'][start:stop], visual), -1), noises[start:stop])
            target = teacher(data['states'][start:stop], noises[start:stop])
            raw_errors.append((predicted - target).square().cpu())
            clip_errors.append((predicted.clamp(-1, 1) - target.clamp(-1, 1)).square().cpu())
            xyz_errors.append((visual[:, :3] - data['states'][start:stop, 19:22]).square().cpu())
        raw, clipped, xyz = map(torch.cat, (raw_errors, clip_errors, xyz_errors))
        if not all(torch.isfinite(error).all().item() for error in (raw, clipped, xyz)):
            raise ValueError('Nonfinite validation error.')
        model.train()
        return {'raw_mse': raw.mean().item(), 'clipped_mse': clipped.mean().item(),
                'raw_mse_by_action': raw.mean(0).tolist(), 'clipped_mse_by_action': clipped.mean(0).tolist(),
                'xyz_axis_rmse_cm': (xyz.mean(0).sqrt() * 10).tolist(),
                'xyz_euclidean_rmse_cm': (xyz.sum(1).mean().sqrt() * 10).item(), 'image_mode': mode, 'samples': len(noises)}

    def save(step):
        samplers = SimpleNamespace(bit_generator=SimpleNamespace(state={'old': old_rng.bit_generator.state, 'new': new_rng.bit_generator.state}))
        save_student_checkpoint(args.run, step, model, optimizer, config, samplers, identity,
                                old_val['images'][:4], old_val['proprio'][:4], val_noises['old'][:4])

    segment = {'start_step': args.parent_step, 'additional_updates': args.steps, 'teacher': identity,
               'runtime': runtime_record(device), 'source_sha256': visual_hashes(), 'additional_source_sha256': extra_sources,
               'initial_validation': {name: validate(name) for name in val_sets}, 'first_update': {}, 'validations': []}
    # All construction, loading, validation generators, and initial checks are done.
    restore_rng(parent['rng'], device)
    model.train()
    started = last_record = time.perf_counter()
    last_update = 0
    with (args.run / 'training.jsonl').open('x') as stream:
        for update in range(1, args.steps + 1):
            common = old_rng.integers(0, len(old['images']), 128)
            first, second = (pool[new_rng.integers(0, len(pool), 64)] for pool in pools)
            common, first, second = (torch.as_tensor(index, device=device) for index in (common, first, second))
            images, proprio, states = (torch.cat((old[key][common], new[key][first], new[key][second]), 0)
                                       for key in ('images', 'proprio', 'states'))
            noises = torch.randn((256, 5), device=device)
            with torch.no_grad():
                targets = teacher(states, noises)
            optimizer.zero_grad(set_to_none=True)
            visual = model.encoder(images.float() / 255)
            predicted = model.actor(torch.cat((proprio, visual), -1), noises)
            action_mse = (predicted - targets).square().mean()
            xyz_mse = (visual[:, :3] - states[:, 19:22]).square().mean()
            loss = action_mse + xyz_mse
            loss.backward()
            if update == 1:
                encoder_before = model.encoder[0].weight.detach().clone()
                segment['first_update'] = {'loss': loss.item(), 'all_gradients_finite': all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()),
                    'encoder_gradient_norm': model.encoder[0].weight.grad.norm().item(),
                    'teacher_gradients_absent': all(p.grad is None and not p.requires_grad for p in teacher.parameters())}
                if not segment['first_update']['all_gradients_finite']:
                    raise ValueError('Nonfinite first correction update.')
            optimizer.step()
            if update == 1:
                segment['first_update']['encoder_max_abs_parameter_change'] = (model.encoder[0].weight - encoder_before).abs().max().item()
            if (update <= 1000 and update % 250 == 0) or update % 1000 == 0 or update == args.steps:
                synchronize(device)
                now = time.perf_counter()
                row = {'update': update, 'step': args.parent_step + update, 'loss': loss.item(), 'action_mse': action_mse.item(), 'xyz_mse': xyz_mse.item(),
                       'updates_per_second': (update - last_update) / (now - last_record), 'elapsed_seconds': now - started,
                       'teacher_parameters_unchanged': teacher_digest() == initial_teacher_digest,
                       'all_parameters_finite': all(torch.isfinite(p).all().item() for p in model.parameters())}
                if not np.isfinite([row['loss'], row['action_mse'], row['xyz_mse']]).all() or not row['teacher_parameters_unchanged'] or not row['all_parameters_finite']:
                    raise ValueError('Nonfinite correction record or changed teacher.')
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                print(json.dumps(row), flush=True)
                last_record, last_update = now, update
            if update == 1000:
                seconds = time.perf_counter() - started
                segment['first_1000'] = {'training_seconds': seconds, 'updates_per_second': 1000 / seconds, 'includes': 'first-step/record checks; excludes checkpoint write'}
                segment['first_1000']['projected20k_seconds'] = seconds * 20
                if update < args.steps:
                    save(args.parent_step + update)
                if seconds * 20 > 1800:
                    raise RuntimeError('Real first1000 projects over30min/arm; early checkpoint retained.')
            if update % 5000 == 0 or update == args.steps:
                result = {'step': args.parent_step + update, 'update': update, 'normal': {name: validate(name) for name in val_sets}}
                segment['validations'].append(result)
                print(json.dumps({'validation': result}), flush=True)
        synchronize(device)
        training_seconds = time.perf_counter() - started
    save(args.parent_step + args.steps)
    segment.update(completed_step=args.parent_step + args.steps, training_seconds=training_seconds,
                   mean_updates_per_second=args.steps / training_seconds, memory=memory(device),
                   final_validation={name: {mode: validate(name, mode) for mode in ('normal', 'shuffle', 'zero')} for name in val_sets},
                   teacher_initial_parameter_sha256=initial_teacher_digest,
                   teacher_parameters_unchanged=teacher_digest() == initial_teacher_digest, environment_steps_added=0)
    if not segment['teacher_parameters_unchanged']:
        raise ValueError('Frozen teacher changed at completion.')
    write_json(args.run / f'segment_{args.parent_step}_{args.parent_step + args.steps}.json', segment, exclusive=True)
    print(json.dumps({'completed_step': segment['completed_step'], 'first_1000': segment.get('first_1000'), 'final_validation': segment['final_validation']}), flush=True)


if __name__ == '__main__':
    main()
