#!/usr/bin/env python3
"""Build/load the fixed five-snapshot, inference-only FQL student average."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '复现/fql/pytorch'))
SCHEMA = 'inference_only_student_average_v2'
ENDPOINTS = (800000, 900000, 1000000)
WINDOW = 5
ENV_NAME = 'cube-single-play-singletask-v0'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def config_digest(config):
    return hashlib.sha256(canonical(config).encode('utf-8')).hexdigest()


def check_config(config):
    if config['seed'] not in (0, 1, 2) or config['env_name'] != ENV_NAME:
        raise ValueError('Expected a native PyTorch seed 0/1/2 task-2 source chain.')
    c = config['agent']
    if tuple(c['ob_dims']) != (28,) or c['action_dim'] != 5 or c['encoder'] is not None:
        raise ValueError('Expected the frozen state-only 28-state / 5-action architecture.')
    if config['training_device'] != 'mps' or not config.get('data_manifest'):
        raise ValueError('Expected the frozen MPS training and dataset identities.')


def student(config, state, device='cpu'):
    import torch
    from fql_torch.networks import ActorVectorField
    c = config['agent']
    policy = ActorVectorField(28, 5, c['actor_hidden_dims'], c['actor_layer_norm'], with_time=False)
    if any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in policy.modules()):
        raise ValueError('This averaging protocol excludes BatchNorm state.')
    for name, value in state.items():
        if value.dtype != torch.float32 or value.device.type != 'cpu' or not torch.isfinite(value).all():
            raise ValueError(f'Expected finite CPU float32 student parameter {name}.')
    policy.load_state_dict(state, strict=True)
    return policy.to(device=device, dtype=torch.float32).eval().requires_grad_(False)


def build(run_dir, epoch, output_dir):
    import torch
    from fql_torch.runtime import configure_device, sha256, source_hashes, write_json
    configure_device('cpu', 4)
    run_dir = Path(run_dir).resolve()
    epochs = list(range(epoch - 40000, epoch + 1, 10000))
    config_path = run_dir / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    check_config(config)
    if epoch not in ENDPOINTS:
        raise ValueError('Only predeclared 800k/900k/1M endpoints are allowed.')
    core_sha = source_hashes()
    sums, shapes, sources = {}, {}, []
    prefix = 'actor_onestep_flow.'
    for source_epoch in epochs:
        path = run_dir / f'checkpoint_{source_epoch}.pt'
        marker_path = run_dir / f'checkpoint_{source_epoch}.json'
        marker = json.loads(marker_path.read_text(encoding='utf-8'))
        digest = sha256(path)
        if (marker['schema_version'] != 1 or marker['completed_updates'] != source_epoch
                or marker['checkpoint_file'] != path.name or marker['sha256'] != digest
                or marker['bytes'] != path.stat().st_size or marker['source_sha256'] != core_sha
                or marker['training_device'] != config['training_device']):
            raise ValueError(f'Uncommitted or mismatched source checkpoint {source_epoch}.')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if (payload['schema_version'] != 1 or payload['completed_updates'] != source_epoch
                or canonical(payload['config']) != canonical(config)
                or payload['source_sha256'] != core_sha
                or payload['training_device'] != config['training_device']):
            raise ValueError(f'Source seed/configuration/data/device/core differs at {source_epoch}.')
        state = {name[len(prefix):]: value for name, value in payload['model'].items()
                 if name.startswith(prefix)}
        current_shapes = {name: tuple(value.shape) for name, value in state.items()}
        if not state or (shapes and current_shapes != shapes):
            raise ValueError(f'Student keys or architecture differ at {source_epoch}.')
        shapes = current_shapes
        for name, value in state.items():
            if value.dtype != torch.float32 or value.device.type != 'cpu' or not torch.isfinite(value).all():
                raise ValueError(f'Invalid student parameter {name} at {source_epoch}.')
            if name not in sums:
                sums[name] = value.detach().to(dtype=torch.float64).clone()
            else:
                sums[name].add_(value.detach().to(dtype=torch.float64))
        sources.append({'epoch': source_epoch, 'path': str(path), 'sha256': digest,
                        'marker_sha256': sha256(marker_path)})
        if source_epoch == epoch:
            probe = {name: payload['reload_probe'][name].detach().cpu().clone()
                     for name in ('observations', 'noises')}
        del state, payload
    averaged = {name: (value / WINDOW).to(dtype=torch.float32) for name, value in sums.items()}
    policy = student(config, averaged)
    if tuple(probe['observations'].shape) != (4, 28) or tuple(probe['noises'].shape) != (4, 5):
        raise ValueError('Unexpected endpoint reload-probe input shapes.')
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in probe.values()):
        raise ValueError('Expected finite float32 endpoint reload-probe inputs.')
    with torch.inference_mode():
        probe['actions'] = policy(probe['observations'], probe['noises']).clamp(-1, 1).cpu().clone()
    if not torch.isfinite(probe['actions']).all():
        raise ValueError('Invalid averaged-policy reload-probe actions.')
    identity = {'schema': SCHEMA, 'inference_only': True, 'resumable': False,
        'artifact_file': 'student_policy.pt', 'policy_label': 'average5', 'window': WINDOW,
        'method': 'equal_weight_parameter_average', 'span': [epochs[0], epochs[-1]],
        'source_run_dir': str(run_dir), 'source_epochs': epochs, 'weights': [0.2] * WINDOW,
        'source_checkpoints': sources, 'source_config_sha256': sha256(config_path),
        'source_config_canonical_sha256': config_digest(config), 'frozen_core_sha': core_sha,
        'generation_script_sha256': sha256(Path(__file__)), 'training_seed': config['seed'],
        'completed_updates': epoch, 'max_source_updates': epoch, 'probe_source_epoch': epoch,
        'additional_optimizer_updates': 0, 'averaged_component': 'actor_onestep_flow only',
        'accumulation': 'CPU float64 sum in ascending epoch order, divide by 5, cast to float32',
        'excluded_state': ['Critic', 'teacher', 'Adam', 'training RNG', 'data sampler'],
        'scope': 'Fixed-window deployment comparison; cannot resume training.'}
    artifact = {**identity, 'source_config': config, 'student': averaged, 'reload_probe': probe}
    path = output_dir / identity['artifact_file']
    temporary = path.with_suffix('.pt.tmp')
    with temporary.open('xb') as stream:
        torch.save(artifact, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    marker = {**identity, 'sha256': sha256(path), 'bytes': path.stat().st_size}
    write_json(output_dir / 'student_policy.json', marker, exclusive=True)
    write_json(output_dir / 'config.json', {**identity, 'source_config': config,
        'marker_file': 'student_policy.json', 'artifact_sha256': marker['sha256']}, exclusive=True)
    print(json.dumps({'status': 'completed', 'training_seed': config['seed'], 'epoch': epoch,
                      'sha256': marker['sha256'], 'output_dir': str(output_dir)}), flush=True)


def load(policy_dir, device='cpu'):
    """Return (student module, artifact, marker, reload check); no Adam/RNG restoration."""
    import torch
    from fql_torch.runtime import configure_device, sha256, source_hashes
    device = configure_device(device, 4)
    policy_dir = Path(policy_dir).resolve()
    path = policy_dir / 'student_policy.pt'
    marker = json.loads((policy_dir / 'student_policy.json').read_text(encoding='utf-8'))
    if marker['sha256'] != sha256(path) or marker['bytes'] != path.stat().st_size:
        raise ValueError('Inference artifact SHA/size differs from its commit marker.')
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    if any(artifact.get(key) != value for key, value in marker.items() if key not in ('sha256', 'bytes')):
        raise ValueError('Inference artifact and marker identities differ.')
    if (artifact['schema'] != SCHEMA or not artifact['inference_only'] or artifact['resumable']
            or artifact['window'] != WINDOW or artifact['additional_optimizer_updates'] != 0
            or any(key in artifact for key in ('optimizer', 'rng', 'sampler', 'model'))):
        raise ValueError('Expected a committed nonresumable five-snapshot student policy.')
    epoch = artifact['completed_updates']
    if (epoch not in ENDPOINTS or artifact['max_source_updates'] != epoch
            or artifact['source_epochs'] != list(range(epoch - 40000, epoch + 1, 10000))
            or artifact['weights'] != [0.2] * WINDOW or artifact['policy_label'] != 'average5'):
        raise ValueError('Artifact differs from the predeclared endpoint/window rule.')
    config = artifact['source_config']
    check_config(config)
    if (artifact['training_seed'] != config['seed']
            or config_digest(config) != artifact['source_config_canonical_sha256']
            or artifact['frozen_core_sha'] != source_hashes()
            or artifact['generation_script_sha256'] != sha256(Path(__file__))):
        raise ValueError('Inference artifact seed/configuration/core/generation identity differs.')
    policy = student(config, artifact['student'], device=device)
    probe = artifact['reload_probe']
    for name, shape in (('observations', (4, 28)), ('noises', (4, 5)), ('actions', (4, 5))):
        if (tuple(probe[name].shape) != shape or probe[name].dtype != torch.float32
                or not torch.isfinite(probe[name]).all()):
            raise ValueError(f'Invalid reload-probe {name}.')
    with torch.inference_mode():
        actual = policy(probe['observations'].to(device), probe['noises'].to(device)).clamp(-1, 1).cpu()
    atol, rtol = (1e-6, 0.0) if device.type == 'cpu' else (1e-4, 1e-4)
    torch.testing.assert_close(actual, probe['actions'], atol=atol, rtol=rtol)
    check = {'passed': True, 'completed_updates': epoch, 'training_seed': config['seed'],
        'maximum_absolute_action_difference': float((actual - probe['actions']).abs().max()),
        'atol': atol, 'rtol': rtol, 'evaluation_device': str(device),
        'policy_sha256': marker['sha256'], 'inference_only': True, 'resumable': False}
    return policy, artifact, marker, check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--epoch', type=int, choices=ENDPOINTS, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        build(args.run_dir, args.epoch, output_dir)
    except BaseException as error:
        with (output_dir / 'error.json').open('x', encoding='utf-8') as stream:
            json.dump({'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
                       'traceback': traceback.format_exc(), 'command': [sys.executable, *sys.argv]},
                      stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        raise


if __name__ == '__main__':
    main()
