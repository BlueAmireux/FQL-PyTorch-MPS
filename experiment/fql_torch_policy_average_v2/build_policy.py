#!/usr/bin/env python3
"""Build an inference-only equal-weight average of the frozen seed-0 student.

No training, dataset loading, environment interaction or MPS execution occurs.
The output has no optimizer/RNG state and cannot resume FQL training.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[2]
PYTORCH = ROOT / '复现' / 'fql' / 'pytorch'
RUN = ROOT / 'experiment' / 'fql_torch_migration' / 'random_mps1000'
SCHEMA = 'inference_only_student_average_v1'
sys.path.insert(0, str(PYTORCH))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window', type=int, choices=(5, 10), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def build(args, output_dir):
    import torch
    from fql_torch.agent import FQLAgent
    from fql_torch.checkpoint import load_checkpoint
    from fql_torch.runtime import configure_device, sha256, source_hashes, write_json

    configure_device('cpu', cpu_threads=4)
    epochs = list(range(1000000 - (args.window - 1) * 10000, 1000001, 10000))
    reference_config = json.loads((RUN / 'config.json').read_text(encoding='utf-8'))
    if reference_config['seed'] != 0 or reference_config['env_name'] != 'cube-single-play-singletask-v0':
        raise ValueError('Expected the frozen native PyTorch seed-0 cube-single chain.')
    if tuple(reference_config['agent']['ob_dims']) != (28,) or reference_config['agent']['action_dim'] != 5:
        raise ValueError('Expected the frozen 28-state / 5-action architecture.')
    if not reference_config.get('data_manifest'):
        raise ValueError('Source config must retain the frozen dataset identity.')
    core_sha = source_hashes()
    sums, shapes, source_checkpoints = {}, {}, []
    for epoch in epochs:
        agent, config, payload = load_checkpoint(RUN, epoch, device='cpu', restore_rng=False)
        if json.dumps(config, sort_keys=True) != json.dumps(reference_config, sort_keys=True) or payload['source_sha256'] != core_sha:
            raise ValueError(f'Configuration, dataset identity or core source differs at epoch {epoch}.')
        if payload['training_device'] != reference_config['training_device']:
            raise ValueError(f'Training-device identity differs at epoch {epoch}.')
        if any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
               for module in agent.actor_onestep_flow.modules()):
            raise ValueError('This averaging protocol excludes BatchNorm state.')
        student = agent.actor_onestep_flow.state_dict()
        current_shapes = {name: tuple(value.shape) for name, value in student.items()}
        if shapes and current_shapes != shapes:
            raise ValueError(f'Student keys or architecture differ at epoch {epoch}.')
        shapes = current_shapes
        for name, value in student.items():
            if value.dtype != torch.float32 or value.device.type != 'cpu' or not torch.isfinite(value).all():
                raise ValueError(f'Expected finite CPU float32 student parameter {name} at epoch {epoch}.')
            if name not in sums:
                sums[name] = value.detach().to(dtype=torch.float64).clone()
            else:
                sums[name].add_(value.detach().to(dtype=torch.float64))
        checkpoint_path = RUN / f'checkpoint_{epoch}.pt'
        marker_path = RUN / f'checkpoint_{epoch}.json'
        source_checkpoints.append({'epoch': epoch, 'path': str(checkpoint_path),
                                   'sha256': sha256(checkpoint_path),
                                   'marker_sha256': sha256(marker_path)})
        if epoch == 1000000:
            probe = {name: torch.as_tensor(payload['reload_probe'][name], dtype=torch.float32,
                                          device='cpu').detach().clone()
                     for name in ('observations', 'noises')}
        del student, agent, payload, config
    averaged = {name: (value / args.window).to(dtype=torch.float32) for name, value in sums.items()}
    if not all(torch.isfinite(value).all() for value in averaged.values()):
        raise ValueError('Averaged student parameters are nonfinite.')
    if tuple(probe['observations'].shape) != (4, 28) or tuple(probe['noises'].shape) != (4, 5):
        raise ValueError('Expected the source 1M checkpoint reload-probe input shapes.')
    if not all(torch.isfinite(value).all() for value in probe.values()):
        raise ValueError('Source reload-probe inputs are nonfinite.')

    # Validate the naked student state against an independently created FQL agent.
    # Its randomly initialized Critic/teacher and Adam are never saved or used.
    averaged_agent = FQLAgent(reference_config['agent'], device='cpu', seed=0)
    averaged_agent.actor_onestep_flow.load_state_dict(averaged, strict=True)
    averaged_agent.eval()
    with torch.inference_mode():
        probe['actions'] = averaged_agent.sample_actions(probe['observations'],
                                                        noises=probe['noises']).detach().cpu().clone()
    if tuple(probe['actions'].shape) != (4, 5) or not torch.isfinite(probe['actions']).all():
        raise ValueError('Averaged student generated invalid reload-probe actions.')
    del averaged_agent, sums
    identity = {'schema': SCHEMA, 'inference_only': True, 'resumable': False,
                'artifact_file': 'student_policy.pt', 'window': args.window,
                'method': 'equal_weight_parameter_average', 'span': [epochs[0], epochs[-1]],
                'source_run_dir': str(RUN), 'source_epochs': epochs,
                'weights': [1.0 / args.window] * args.window,
                'source_checkpoints': source_checkpoints,
                'source_config_sha256': sha256(RUN / 'config.json'),
                'frozen_core_sha': core_sha, 'generation_script_sha256': sha256(Path(__file__)),
                'additional_optimizer_updates': 0, 'max_source_updates': 1000000,
                'training_seed': 0, 'probe_source_epoch': 1000000,
                'averaged_component': 'actor_onestep_flow only; Critic and teacher are excluded',
                'accumulation': 'CPU float64 sum in ascending epoch order, divide equally, cast to float32',
                'scope': 'development-only inference artifact; not a resumable checkpoint or final improvement claim'}
    artifact = {**identity, 'source_config': reference_config, 'student': averaged, 'reload_probe': probe}
    path = output_dir / identity['artifact_file']
    temporary = path.with_suffix('.pt.tmp')
    with temporary.open('xb') as stream:
        torch.save(artifact, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    marker = {**identity, 'sha256': sha256(path), 'bytes': path.stat().st_size}
    write_json(output_dir / 'student_policy.json', marker, exclusive=True)
    write_json(output_dir / 'config.json', {
        'schema': SCHEMA, 'inference_only': True, 'resumable': False,
        'artifact_file': identity['artifact_file'], 'marker_file': 'student_policy.json',
        'artifact_sha256': marker['sha256'], 'window': args.window,
        'source_config': reference_config, 'source_epochs': epochs,
        'additional_optimizer_updates': 0, 'max_source_updates': 1000000}, exclusive=True)
    print(json.dumps({'status': 'completed', 'window': args.window, 'sha256': marker['sha256'],
                      'output_dir': str(output_dir), 'additional_optimizer_updates': 0}), flush=True)


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        build(args, output_dir)
    except BaseException as error:
        with (output_dir / 'error.json').open('x', encoding='utf-8') as stream:
            json.dump({'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
                       'traceback': traceback.format_exc(), 'command': [sys.executable, *sys.argv]},
                      stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        raise


if __name__ == '__main__':
    main()
