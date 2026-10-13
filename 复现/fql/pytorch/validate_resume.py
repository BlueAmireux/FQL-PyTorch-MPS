#!/usr/bin/env python3
"""Check strict same-device continuation on archived real transitions.

Runs continuous 20 updates versus 10 + checkpoint + fresh-process 10. This is a
short state-recovery check, not a long-run stability or performance experiment.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--batch', type=Path, default=ROOT / 'experiment/fql_state_setup/inspection/real_batch.npz')
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--optimizer-impl', choices=('foreach', 'fused'), default='foreach')
    parser.add_argument('--compile', action='store_true', help='Use the agent static loss/backward compilation API.')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--worker', choices=('reference', 'first', 'second'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cpu_threads < 1 or not 0 <= args.seed < 2**32:
        parser.error('Threads must be positive and seed must fit uint32.')
    if args.output_dir is None:
        suffix = '_compile' if args.compile else ''
        args.output_dir = ROOT / f'experiment/fql_torch_migration/resume_{args.device}_{args.optimizer_impl}{suffix}'
    return args


def run_worker(args):
    import numpy as np
    import torch
    from fql_torch.agent import FQLAgent, get_config
    from fql_torch.checkpoint import capture_rng, cpu_tree, load_checkpoint, save_checkpoint
    from fql_torch.data import FIELDS, ResidentDataset, TASK
    from fql_torch.runtime import configure_device, runtime_record, sha256, synchronize, write_json

    device = configure_device(args.device, args.cpu_threads)
    output_dir = args.output_dir.resolve()
    run_dir = output_dir / ('reference' if args.worker == 'reference' else 'split')
    with np.load(args.batch, allow_pickle=False) as fixture:
        arrays = {name: np.array(fixture[name], dtype=np.float32, copy=True) for name in FIELDS}
        original_indices = np.array(fixture['indices'], dtype=np.int64, copy=True)
    if arrays['observations'].shape != (256, 28) or arrays['actions'].shape != (256, 5):
        raise ValueError('Resume check requires the archived real 256-transition fixture.')
    probe = arrays['observations'][:4].copy()
    data = ResidentDataset(arrays, device=device, placement='device', seed=args.seed)
    manifest = {'path': str(args.batch.resolve()), 'sha256': sha256(args.batch),
                'fixture_transitions': 256, 'source': 'official OGBench train sampled real batch'}
    if args.worker == 'second':
        agent, config, payload = load_checkpoint(run_dir, 10, device=device, restore_rng=True)
        if (config['cpu_threads'] != args.cpu_threads or config['fixture'] != manifest
                or config['training_device'] != args.device
                or config['agent']['optimizer_impl'] != args.optimizer_impl
                or config.get('compile', False) != args.compile):
            raise ValueError('Resume protocol changed.')
        data.restore_sampler(payload['sampler'])
        start, end = 10, 20
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        agent_config = get_config()
        agent_config['optimizer_impl'] = args.optimizer_impl
        config = {'env_name': TASK, 'seed': args.seed, 'agent': agent_config,
                  'training_device': args.device, 'cpu_threads': args.cpu_threads, 'fixture': manifest,
                  'compile': args.compile,
                  'purpose': 'Strict fresh-process same-device state continuation diagnostic.'}
        agent = FQLAgent(config['agent'], device=device, seed=args.seed)
        write_json(run_dir / 'config.json', config, exclusive=True)
        save_checkpoint(run_dir, 0, agent, config, data.sampler_state(), probe)
        start, end = 0, 20 if args.worker == 'reference' else 10

    if args.compile:
        agent.configure_loss_compile(True)

    rows = []
    with (run_dir / f'updates_{start}_to_{end}.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = None
        for update in range(start + 1, end + 1):
            batch, indices = data.sample(config['agent']['batch_size'], return_indices=True)
            metrics = agent.update(batch)
            values = {key: float(value.cpu()) for key, value in metrics.items() if '/' not in key}
            if not np.isfinite(list(values.values())).all():
                raise FloatingPointError(f'Nonfinite metric at update {update}.')
            row = {'update': update,
                   'sampled_fixture_rows_sha256': hashlib.sha256(indices.tobytes()).hexdigest(),
                   'sampled_original_indices_sha256': hashlib.sha256(original_indices[indices].tobytes()).hexdigest(),
                   **values}
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            rows.append(row)
            # Identical save points verify that saving does not advance training RNG.
            if update in (10, 20):
                save_checkpoint(run_dir, update, agent, config, data.sampler_state(), probe)

    synchronize(device)
    snapshot = {'completed_updates': agent.updates, 'model': cpu_tree(agent.state_dict()),
                'optimizer': cpu_tree(agent.optimizer.state_dict()), 'rng': capture_rng(device),
                'sampler': cpu_tree(data.sampler_state())}
    next_batch, next_indices = data.sample(config['agent']['batch_size'], return_indices=True)
    snapshot['next_batch'] = cpu_tree(next_batch)
    snapshot['next_indices'] = torch.from_numpy(next_indices.copy())
    snapshot['next_torch_random'] = torch.randn(16)
    snapshot['next_device_random'] = torch.randn(16, device=device).cpu()
    path = run_dir / f'snapshot_{end}.pt'
    with path.open('xb') as stream:
        torch.save(snapshot, stream)
    write_json(run_dir / f'worker_{args.worker}.json', {
        'status': 'completed', 'start': start, 'end': end, 'updates': len(rows),
        'runtime': runtime_record(device), 'snapshot': path.name, 'snapshot_sha256': sha256(path),
        'fixture': manifest}, exclusive=True)


def compare_tree(left, right, path, differences, counts):
    import torch
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        counts['tensor_leaves'] += 1
        same = (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right))
        if not same:
            item = {'path': path, 'reason': 'tensor differs'}
            if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and left.shape == right.shape:
                item['max_absolute_difference'] = float((left.to(torch.float64) - right.to(torch.float64)).abs().max())
            differences.append(item)
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            differences.append({'path': path, 'reason': 'dictionary keys differ'})
        for key in left.keys() & right.keys():
            compare_tree(left[key], right[key], f'{path}/{key}', differences, counts)
        return
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            differences.append({'path': path, 'reason': 'sequence type/length differs'})
        for index, (a, b) in enumerate(zip(left, right)):
            compare_tree(a, b, f'{path}/{index}', differences, counts)
        return
    counts['scalar_leaves'] += 1
    if type(left) is not type(right) or left != right:
        differences.append({'path': path, 'reason': 'scalar differs', 'reference': left, 'resumed': right})


def controller(args):
    import torch
    from fql_torch.runtime import sha256, write_json

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    command_base = [sys.executable, str(Path(__file__).resolve()),
                    '--output-dir', str(output_dir), '--batch', str(args.batch.resolve()),
                    '--device', args.device, '--optimizer-impl', args.optimizer_impl,
                    '--cpu-threads', str(args.cpu_threads), '--seed', str(args.seed)]
    if args.compile:
        command_base.append('--compile')
    subprocess_records = []
    for mode in ('reference', 'first', 'second'):
        command = [*command_base, '--worker', mode]
        with (output_dir / f'{mode}.log').open('x', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        subprocess_records.append({'mode': mode, 'command': command, 'exit_code': result.returncode})
        if result.returncode:
            write_json(output_dir / 'comparison.json', {'status': 'failed', 'passed': False,
                'stage': mode, 'processes': subprocess_records}, exclusive=True)
            raise RuntimeError(f'{mode} worker exited {result.returncode}; evidence kept in {output_dir}.')

    reference_path = output_dir / 'reference/snapshot_20.pt'
    split_path = output_dir / 'split/snapshot_20.pt'
    reference = torch.load(reference_path, map_location='cpu', weights_only=True)
    resumed = torch.load(split_path, map_location='cpu', weights_only=True)
    differences, counts = [], {'tensor_leaves': 0, 'scalar_leaves': 0}
    compare_tree(reference, resumed, 'state', differences, counts)
    with (output_dir / 'reference/updates_0_to_20.csv').open(newline='') as stream:
        reference_rows = list(csv.DictReader(stream))[10:]
    with (output_dir / 'split/updates_10_to_20.csv').open(newline='') as stream:
        resumed_rows = list(csv.DictReader(stream))
    rows_equal = reference_rows == resumed_rows
    if not rows_equal:
        differences.append({'path': 'updates_11_to_20', 'reason': 'sample indices or metric rows differ'})
    report = {'status': 'completed', 'passed': not differences,
              'protocol': f'fresh-process continuous {args.device} 20 vs {args.device} 10/save/load/10; 4x512 batch256 float32',
              'fixture': {'path': str(args.batch.resolve()), 'sha256': sha256(args.batch)},
              'training_seed': args.seed, 'cpu_threads': args.cpu_threads,
              'device': args.device, 'optimizer_impl': args.optimizer_impl,
              'compile': args.compile,
              'torch_cpu_rng_checked': True,
              'torch_mps_rng_checked': args.device == 'mps' and reference['rng']['mps'] is not None,
              'model_adam_rng_sampler_next_batch_exact': not differences,
              'update_11_to_20_batch_and_metrics_exact': rows_equal,
              'checked_leaves': counts, 'differences': differences,
              'processes': subprocess_records,
              'reference_snapshot_sha256': sha256(reference_path), 'resumed_snapshot_sha256': sha256(split_path),
              'scope': f'Strict short {args.device} continuation only; does not establish another backend, cross-framework or long-run equivalence.',
              'online_training_environment_steps': 0,
              'full_offline_dataset_loaded': False,
              'fixture_sampling': 'Uniform with replacement from 256 archived real transitions, not full training.'}
    write_json(output_dir / 'comparison.json', report, exclusive=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if differences:
        raise RuntimeError('Strict continuation comparison failed; evidence retained.')


def main():
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        try:
            controller(args)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
