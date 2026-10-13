#!/usr/bin/env python3
"""State-only offline FQL training on Apple MPS; explicit, resumable updates."""
import argparse
import csv
import json
from pathlib import Path
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, default=ROOT / 'DataSet/fql')
    p.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--updates', type=int, default=1000, help='ADDITIONAL optimizer updates in this invocation.')
    p.add_argument('--resume-epoch', type=int, help='Explicit committed checkpoint epoch to resume, same device.')
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--checkpoint-every', type=int, default=10000)
    p.add_argument('--data-placement', choices=('device', 'cpu'), default='device')
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--optimizer-impl', choices=('foreach', 'fused'), default='foreach',
                   help='Implementation of the same Adam equations; fixed in run config.')
    p.add_argument('--invocation-id', default='', help='Optional short label for a retry; keeps previous logs.')
    p.add_argument('--trace-batches', action='store_true', help='Write each sampled index SHA for short resume validation.')
    p.add_argument('--compile', action='store_true', help='Experimental loss compilation; benchmark and verify separately.')
    args = p.parse_args()
    if min(args.updates, args.log_every, args.checkpoint_every, args.cpu_threads) < 1:
        p.error('Updates, intervals and CPU threads must be positive.')
    if not 0 <= args.seed < 2**32 or (args.resume_epoch is not None and args.resume_epoch < 0):
        p.error('Seed must fit uint32; resume epoch must be nonnegative.')
    if args.invocation_id and (len(args.invocation_id) > 40 or not all(c.isalnum() or c in '-_' for c in args.invocation_id)):
        p.error('--invocation-id must be a short alphanumeric/hyphen/underscore label.')
    return args


def main():
    args = parse_args()
    import hashlib
    import numpy as np
    import torch
    from fql_torch.agent import FQLAgent, get_config
    from fql_torch.checkpoint import load_checkpoint, save_checkpoint
    from fql_torch.data import load_official, ResidentDataset, TASK
    from fql_torch.runtime import configure_device, memory, runtime_record, synchronize, write_json
    device = configure_device(args.device, args.cpu_threads)
    run_dir = args.run_dir.resolve()
    if args.resume_epoch is None:
        run_dir.mkdir(parents=True, exist_ok=False)
        config = {'env_name': TASK, 'seed': args.seed, 'agent': get_config(),
                  'training_device': args.device, 'data_dir': str(args.data_dir.resolve()),
                  'data_placement': args.data_placement, 'compile': args.compile,
                  'cpu_threads': args.cpu_threads,
                  'purpose': 'State offline FQL; limited runs do not establish paper scores.'}
        config['agent']['optimizer_impl'] = args.optimizer_impl
        agent = FQLAgent(config['agent'], device=device, seed=args.seed)
        start_epoch = 0
        payload = None
    else:
        agent, config, payload = load_checkpoint(run_dir, args.resume_epoch, device=device, restore_rng=True)
        if args.compile != config.get('compile', False):
            raise ValueError('Resume must preserve compile mode.')
        if args.cpu_threads != config['cpu_threads']:
            raise ValueError('Resume must preserve the recorded CPU thread count.')
        if args.optimizer_impl != config['agent']['optimizer_impl']:
            raise ValueError('Resume must preserve the recorded optimizer implementation.')
        start_epoch = args.resume_epoch
    target = start_epoch + args.updates
    suffix = f'_{args.invocation_id}' if args.invocation_id else ''
    log_path = run_dir / f'train_{start_epoch}_to_{target}{suffix}.csv'
    summary_path = run_dir / f'summary_{start_epoch}_to_{target}{suffix}.json'
    runtime_path = run_dir / f'runtime_{start_epoch}_to_{target}{suffix}.json'
    if any(p.exists() for p in (log_path, summary_path, runtime_path)):
        raise FileExistsError('This invocation already has evidence; never overwrite it.')
    wall_start = time.perf_counter()
    try:
        write_json(runtime_path, runtime_record(device), exclusive=True)
        arrays, validation, data_manifest = load_official(config['data_dir'])
        del validation
        data = ResidentDataset(arrays, device, config['data_placement'], config['seed'])
        probe = np.array(arrays['observations'][:4], copy=True)
        del arrays
        if payload is not None:
            data.restore_sampler(payload['sampler'])
            # Data loading is deterministic and never consumes torch randomness.
            if payload['config']['data_manifest'] != data_manifest:
                raise ValueError('Training dataset identity changed since checkpoint.')
        else:
            config['data_manifest'] = data_manifest
            write_json(run_dir / 'config.json', config, exclusive=True)
            save_checkpoint(run_dir, 0, agent, config, data.sampler_state(), probe)
        if args.compile:
            agent.configure_loss_compile()
        stopped = [False]
        def request_stop(signum, frame):
            stopped[0] = True
            print('\nStop requested; saving after this complete update.', flush=True)
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        synchronize(device)
        train_start = time.perf_counter()
        last_report = train_start
        last_epoch = start_epoch
        traces = (run_dir / f'batches_{start_epoch}_to_{target}{suffix}.csv').open('x', newline='') if args.trace_batches else None
        trace_writer = csv.writer(traces) if traces else None
        if trace_writer:
            trace_writer.writerow(['update', 'sampled_indices_sha256'])
        with log_path.open('x', newline='') as stream:
            writer = None
            for epoch in range(start_epoch + 1, target + 1):
                batch, indices = data.sample(config['agent']['batch_size'], return_indices=True)
                metrics = agent.update(batch)
                last_epoch = epoch
                if trace_writer:
                    trace_writer.writerow([epoch, hashlib.sha256(indices.tobytes()).hexdigest()])
                if (epoch - start_epoch) == 1 or epoch % args.log_every == 0 or epoch == target or stopped[0]:
                    synchronize(device)
                    values = {k: float(v.detach().cpu()) for k,v in metrics.items() if '/' not in k}
                    if not np.isfinite(list(values.values())).all():
                        raise FloatingPointError(f'Nonfinite losses at update {epoch}.')
                    now = time.perf_counter()
                    row = {'update': epoch, 'elapsed_seconds': now-train_start,
                           **values, **memory(device)}
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                    print(f'update={epoch}/{target} loss={values["loss"]:.5g} elapsed={now-train_start:.2f}s ' 
                          f'RSS={row["rss_bytes"]/2**30:.2f}GiB', flush=True)
                    last_report = now
                if epoch % args.checkpoint_every == 0 or epoch == target or stopped[0]:
                    save_checkpoint(run_dir, epoch, agent, config, data.sampler_state(), probe)
                if stopped[0]:
                    break
            if traces:
                traces.close()
        synchronize(device)
        train_seconds = time.perf_counter() - train_start
        summary = {'status': 'stopped' if stopped[0] else 'completed',
                   'completed_updates': last_epoch, 'start_epoch': start_epoch,
                   'added_updates': last_epoch-start_epoch, 'requested_end_epoch': target,
                   'training_loop_seconds': train_seconds,
                   'effective_updates_per_second': (last_epoch-start_epoch)/train_seconds,
                   'invocation_wall_seconds': time.perf_counter()-wall_start,
                   'training_environment_steps': 0, 'device': str(device),
                   'memory': memory(device), 'checkpoint': f'checkpoint_{last_epoch}.pt',
                   'scope': 'Single-seed finite-budget engineering run; not paper-score reproduction.'}
        write_json(summary_path, summary, exclusive=True)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except BaseException as error:
        failure = {'status':'failed', 'error_type':type(error).__name__, 'error':str(error),
                   'traceback':traceback.format_exc()}
        if not summary_path.exists():
            write_json(summary_path, failure)
        raise


if __name__ == '__main__':
    main()
