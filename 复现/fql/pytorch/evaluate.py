#!/usr/bin/env python3
"""Independently evaluate a PyTorch FQL checkpoint on OGBench cube task 2.

No optimizer update or offline dataset load is performed. Evaluation uses an
independent CPU noise generator per episode, including when inference uses MPS.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
ENV_NAME = 'cube-single-play-singletask-v0'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--epoch', type=int, required=True, help='Completed updates / checkpoint suffix.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--env-seed-base', type=int, default=1000)
    parser.add_argument('--action-seed-base', type=int, default=2000)
    parser.add_argument('--video', action='store_true', help='Record episode 0, including failure.')
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    if args.epoch < 0 or args.episodes < 1:
        parser.error('--epoch must be nonnegative and --episodes must be positive.')
    if min(args.env_seed_base, args.action_seed_base) < 0:
        parser.error('Seed bases must be nonnegative.')
    if max(args.env_seed_base, args.action_seed_base) + args.episodes > 2**32:
        parser.error('All evaluation seeds must fit uint32.')
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive.')
    return args


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def success_from_info(info, np):
    if 'success' not in info:
        raise KeyError(f"Missing official info['success']; keys={list(info)}")
    value = np.asarray(info['success'])
    if value.shape != () or not np.isfinite(value) or float(value) not in (0.0, 1.0):
        raise ValueError(f"Expected scalar binary info['success']; got {value!r}")
    return bool(value)


def wilson_interval(successes, count):
    z = 1.959963984540054
    fraction = successes / count
    denominator = 1 + z * z / count
    center = (fraction + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def latency_summary(values, np):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError('Cannot summarize an empty completed evaluation.')
    return {'count': int(values.size), 'mean_ms': float(values.mean() * 1000),
            'median_ms': float(np.median(values) * 1000),
            'p95_ms': float(np.quantile(values, 0.95) * 1000),
            'max_ms': float(values.max() * 1000)}


def runtime_record(args, torch):
    packages = {}
    for name in ('numpy', 'torch', 'ogbench', 'gymnasium', 'mujoco', 'imageio', 'imageio-ffmpeg'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    lock = HERE / 'requirements.lock.txt'
    return {'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'machine': platform.machine(),
            'command': [sys.executable, *sys.argv], 'packages': packages,
            'requested_device': args.device, 'evaluation_device': args.device,
            'mps_built': torch.backends.mps.is_built(),
            'mps_available': torch.backends.mps.is_available(),
            'torch_num_threads': torch.get_num_threads(),
            'effective_flags': {name: os.environ.get(name) for name in (
                'PYTORCH_ENABLE_MPS_FALLBACK', 'PYTORCH_MPS_FAST_MATH',
                'PYTORCH_MPS_PREFER_METAL', 'OMP_NUM_THREADS')},
            'evaluation_script_sha256': sha256(Path(__file__)),
            'dependency_lock_path': str(lock),
            'dependency_lock_sha256': sha256(lock) if lock.is_file() else None,
            'offline_dataset_loaded': False, 'optimizer_updates': 0}


def evaluate(args, output_dir):
    # Delayed imports keep --help usable without any ML/environment installation.
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    import ogbench
    from fql_torch.checkpoint import load_checkpoint
    from fql_torch.runtime import configure_device

    device = configure_device(args.device, args.cpu_threads)
    runtime = runtime_record(args, torch)
    run_dir = args.run_dir.resolve()
    checkpoint_path = run_dir / f'checkpoint_{args.epoch}.pt'
    marker_path = run_dir / f'checkpoint_{args.epoch}.json'
    config_path = run_dir / 'config.json'
    reload_check = {'passed': False, 'epoch': args.epoch,
                    'maximum_absolute_action_difference': None,
                    'note': 'Read-only independent checkpoint load; no optimizer update or dataset load.'}
    try:
        started = time.perf_counter()
        agent, config, payload = load_checkpoint(run_dir, args.epoch, device=args.device, restore_rng=False)
        reload_check['checkpoint_restore_seconds'] = time.perf_counter() - started
        if config.get('env_name') != ENV_NAME:
            raise ValueError(f"This evaluator only supports {ENV_NAME}; got {config.get('env_name')}")
        agent_config = config['agent']
        if tuple(agent_config['ob_dims']) != (28,) or agent_config['action_dim'] != 5:
            raise ValueError('Expected the selected 28-state / 5-action architecture.')
        completed_updates = int(payload['completed_updates'])
        if completed_updates != args.epoch:
            raise ValueError(f'Checkpoint updates {completed_updates} != requested {args.epoch}.')
        training_device = payload['training_device']
        if training_device not in ('cpu', 'mps'):
            raise ValueError(f'Unexpected training_device {training_device!r}')
        cross_device = training_device != args.device
        atol, rtol = (1e-4, 1e-4) if cross_device else (1e-6, 0.0)
        runtime.update(training_device=training_device, cross_device_reload=cross_device,
                       reload_atol=atol, reload_rtol=rtol)
        reload_check.update(training_device=training_device, evaluation_device=args.device,
                            cross_device_reload=cross_device, atol=atol, rtol=rtol,
                            checkpoint_sha256=sha256(checkpoint_path), marker_sha256=sha256(marker_path),
                            completed_updates=completed_updates)
        probe = payload['reload_probe']
        observations = torch.as_tensor(probe['observations'], dtype=torch.float32, device=device)
        noises = torch.as_tensor(probe['noises'], dtype=torch.float32, device=device)
        expected = torch.as_tensor(probe['actions'], dtype=torch.float32, device='cpu').numpy()
        if tuple(observations.shape) != (4, 28) or tuple(noises.shape) != (4, 5) or expected.shape != (4, 5):
            raise ValueError('Unexpected probe shapes: expected [4,28] / [4,5] / [4,5].')
        if hasattr(agent, 'eval'):
            agent.eval()
        started = time.perf_counter()
        with torch.inference_mode():
            actual = agent.sample_actions(observations, noises=noises).detach().cpu().numpy()
        reload_check.update(batched_probe_seconds=time.perf_counter() - started,
                            observations_shape=list(observations.shape), noises_shape=list(noises.shape),
                            actions_shape=list(actual.shape), actual_actions_finite=bool(np.isfinite(actual).all()),
                            expected_actions_finite=bool(np.isfinite(expected).all()))
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError('Invalid action shapes or nonfinite values in reload probe.')
        max_difference = float(np.max(np.abs(actual - expected)))
        reload_check['maximum_absolute_action_difference'] = max_difference
        if not np.allclose(actual, expected, rtol=rtol, atol=atol):
            raise ValueError(f'Reload action probe failed: max difference {max_difference}, atol {atol}, rtol {rtol}.')
        reload_check['passed'] = True
    except Exception as error:
        reload_check.update(error_type=type(error).__name__, error=str(error))
        runtime.update(reload_passed=False)
        write_json(output_dir / 'reload_check.json', reload_check)
        write_json(output_dir / 'runtime.json', runtime)
        raise
    runtime.update(reload_passed=True, maximum_absolute_action_difference=max_difference)
    write_json(output_dir / 'reload_check.json', reload_check)
    write_json(output_dir / 'runtime.json', runtime)

    # env_only returns before OGBench's downloader or offline dataset loader.
    env = ogbench.make_env_and_datasets(ENV_NAME, env_only=True)
    writer = None
    video_status = {'requested': args.video, 'selection_rule': 'episode 0 regardless of success',
                    'status': 'not_requested'}
    rows, reset_infos, inference_seconds, environment_seconds = [], [], [], []
    evaluation_started = time.perf_counter()
    try:
        if env.observation_space.shape != (28,) or env.action_space.shape != (5,):
            raise ValueError(f'Unexpected environment spaces: {env.observation_space}, {env.action_space}')
        if not (np.allclose(env.action_space.low, -1) and np.allclose(env.action_space.high, 1)):
            raise ValueError('Expected official normalized [-1, 1] action space.')
        task_id = getattr(env.unwrapped, '_reward_task_id', None)
        if task_id != 2:
            raise ValueError(f'Expected the frozen OGBench default task 2; got {task_id!r}')
        max_episode_steps = getattr(env.spec, 'max_episode_steps', None)
        if max_episode_steps != 200:
            raise ValueError(f'Expected frozen 200-step horizon; got {max_episode_steps!r}')

        def infer(observation, noise):
            with torch.inference_mode():
                observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
                noise_tensor = noise.to(device=device, dtype=torch.float32)
                action = agent.sample_actions(observation_tensor, noises=noise_tensor)
                # Host conversion synchronizes MPS; timing includes conversion.
                return action.detach().cpu().numpy().astype(np.float32, copy=False)

        columns = ['episode', 'env_seed', 'action_seed', 'return', 'length', 'success',
                   'ever_success', 'terminated', 'truncated', 'episode_seconds', 'trajectory']
        with (output_dir / 'episodes.csv').open('x', newline='', encoding='utf-8') as csv_stream:
            csv_writer = csv.DictWriter(csv_stream, fieldnames=columns)
            csv_writer.writeheader()
            first_call_seconds, warmup_seconds = None, []
            for episode in range(args.episodes):
                env_seed, action_seed = args.env_seed_base + episode, args.action_seed_base + episode
                observation, reset_info = env.reset(seed=env_seed)
                observation = np.asarray(observation, dtype=np.float32)
                if observation.shape != (28,) or not np.isfinite(observation).all():
                    raise ValueError('Reset observation has invalid shape or nonfinite values.')
                reset_infos.append(jsonable(reset_info))
                if episode == 0:
                    # Warmup has its own generator and does not consume episode noise.
                    warmup_rng = torch.Generator(device='cpu').manual_seed(2**32 - 1)
                    started = time.perf_counter()
                    infer(observation, torch.randn(5, generator=warmup_rng, dtype=torch.float32))
                    first_call_seconds = time.perf_counter() - started
                    for _ in range(5):
                        noise = torch.randn(5, generator=warmup_rng, dtype=torch.float32)
                        started = time.perf_counter()
                        infer(observation, noise)
                        warmup_seconds.append(time.perf_counter() - started)
                    if args.video:
                        try:
                            import imageio.v2 as imageio
                            writer = imageio.get_writer(output_dir / 'episode_000.mp4', fps=20, macro_block_size=1)
                            writer.append_data(np.asarray(env.render()))
                            video_status.update(status='recording', path='episode_000.mp4', fps=20, frame_skip=1)
                        except Exception as error:
                            video_status.update(status='failed', error=f'{type(error).__name__}: {error}')
                            if writer is not None:
                                writer.close()
                            writer = None
                noise_rng = torch.Generator(device='cpu').manual_seed(action_seed)
                trajectory = {'observations': [observation.copy()], 'actions': [], 'noises': [],
                              'rewards': [], 'successes': [], 'terminated': [], 'truncated': [],
                              'inference_seconds': []}
                episode_started, done = time.perf_counter(), False
                try:
                    while not done:
                        noise = torch.randn(5, generator=noise_rng, dtype=torch.float32)
                        started = time.perf_counter()
                        action = infer(observation, noise)
                        latency = time.perf_counter() - started
                        if action.shape != (5,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
                            raise ValueError('Policy action has invalid shape, nonfinite values, or exceeds [-1, 1].')
                        started = time.perf_counter()
                        next_observation, reward, terminated, truncated, info = env.step(action)
                        environment_seconds.append(time.perf_counter() - started)
                        success = success_from_info(info, np)
                        next_observation = np.asarray(next_observation, dtype=np.float32)
                        if next_observation.shape != (28,) or not np.isfinite(next_observation).all() or not np.isfinite(reward):
                            raise ValueError('Environment transition contains invalid observations or reward.')
                        trajectory['observations'].append(next_observation.copy())
                        trajectory['actions'].append(action.copy())
                        trajectory['noises'].append(noise.numpy().copy())
                        trajectory['rewards'].append(float(reward))
                        trajectory['successes'].append(success)
                        trajectory['terminated'].append(bool(terminated))
                        trajectory['truncated'].append(bool(truncated))
                        trajectory['inference_seconds'].append(latency)
                        inference_seconds.append(latency)
                        observation, done = next_observation, bool(terminated or truncated)
                        if episode == 0 and writer is not None:
                            try:
                                writer.append_data(np.asarray(env.render()))
                            except Exception as error:
                                video_status.update(status='failed', error=f'{type(error).__name__}: {error}')
                                writer.close()
                                writer = None
                        if len(trajectory['actions']) >= max_episode_steps and not done:
                            raise RuntimeError('200-step horizon reached without terminated/truncated; episode is incomplete.')
                except Exception:
                    np.savez_compressed(output_dir / f'episode_{episode:03d}_partial.npz',
                                        **{name: np.asarray(values) for name, values in trajectory.items()},
                                        env_seed=env_seed, action_seed=action_seed)
                    raise
                trajectory_name = f'episode_{episode:03d}.npz'
                np.savez_compressed(output_dir / trajectory_name,
                                    **{name: np.asarray(values) for name, values in trajectory.items()},
                                    env_seed=env_seed, action_seed=action_seed)
                row = {'episode': episode, 'env_seed': env_seed, 'action_seed': action_seed,
                       'return': float(sum(trajectory['rewards'])), 'length': len(trajectory['actions']),
                       'success': int(success), 'ever_success': int(any(trajectory['successes'])),
                       'terminated': bool(terminated), 'truncated': bool(truncated),
                       'episode_seconds': time.perf_counter() - episode_started, 'trajectory': trajectory_name}
                rows.append(row)
                csv_writer.writerow(row)
                csv_stream.flush()
                print(json.dumps(row), flush=True)
                if episode == 0 and writer is not None:
                    writer.close()
                    writer = None
                    video_status['status'] = 'completed'

        sources = {str(path.relative_to(HERE)): sha256(path) for path in sorted((HERE / 'fql_torch').glob('*.py'))}
        write_json(output_dir / 'environment.json', {
            **runtime, 'env_name': ENV_NAME, 'task_id': task_id,
            'env_spec': {'id': env.spec.id, 'max_episode_steps': max_episode_steps,
                         'entry_point': str(env.spec.entry_point), 'kwargs': env.spec.kwargs},
            'env_only': True, 'observation_space': str(env.observation_space),
            'action_space': str(env.action_space), 'reset_infos': reset_infos,
            'checkpoint': str(checkpoint_path), 'checkpoint_sha256': sha256(checkpoint_path),
            'marker_sha256': sha256(marker_path), 'config_sha256': sha256(config_path),
            'pytorch_source_sha256': sources})
        successes = sum(row['success'] for row in rows)
        metrics = {'status': 'completed',
                   'scope': 'fixed-seed short-budget PyTorch migration verification; not original-paper score reproduction',
                   'env_name': ENV_NAME, 'task_id': task_id, 'training_seed': config['seed'],
                   'checkpoint_epoch': args.epoch, 'completed_updates': completed_updates,
                   'checkpoint_sha256': sha256(checkpoint_path),
                   'episodes': args.episodes, 'successes': successes, 'success_rate': successes / args.episodes,
                   'success_rate_wilson_95_percent_ci': wilson_interval(successes, args.episodes),
                   'success_definition': "Final official info['success']; ever_success is separate and never substituted.",
                   'returns': [row['return'] for row in rows],
                   'mean_return': float(np.mean([row['return'] for row in rows])),
                   'lengths': [row['length'] for row in rows],
                   'mean_length': float(np.mean([row['length'] for row in rows])),
                   'evaluation_environment_steps': sum(row['length'] for row in rows),
                   'training_environment_steps': 0,
                   'env_seeds': [row['env_seed'] for row in rows], 'action_seeds': [row['action_seed'] for row in rows],
                   'policy': 'FQL one-step student with per-step Gaussian noise and action clipping',
                   'action_noise_rule': 'Independent torch CPU Generator seeded with action_seed; float32 randn(5) each step.',
                   'action_noise_storage': 'All per-step noises saved in episode NPZ for exact input replay; differs from JAX PRNG stream.',
                   'inference_latency': {'first_unbatched_call_seconds': first_call_seconds,
                                         'warmup_call_seconds': warmup_seconds,
                                         'steady_state': latency_summary(inference_seconds, np),
                                         'timing_boundary': 'observation/noise conversion, sample_actions, synchronized CPU ndarray conversion; env/render excluded'},
                   'environment_step_latency': latency_summary(environment_seconds, np),
                   'evaluation_seconds_including_warmup_rendering': time.perf_counter() - evaluation_started,
                   'video': video_status,
                   'trajectory_selection_rule': 'All completed episodes saved; episode 0 recorded regardless of outcome.',
                   'terminated_count': sum(row['terminated'] for row in rows),
                   'truncated_count': sum(row['truncated'] for row in rows),
                   'limitations': ['One training seed and few episodes do not establish stable performance.',
                                   'JAX and PyTorch Gaussian RNG streams differ; equal seed numbers are not equal action-noise sequences.',
                                   'GPU backend throughput is not an algorithm improvement or task-success gain.']}
        write_json(output_dir / 'metrics.json', metrics)
        print(json.dumps({'successes': successes, 'episodes': args.episodes,
                          'success_rate': metrics['success_rate'], 'output_dir': str(output_dir)}, ensure_ascii=False), flush=True)
    finally:
        if writer is not None:
            writer.close()
        env.close()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    # Even an empty pre-existing directory is not silently reused.
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        evaluate(args, output_dir)
    except Exception as error:
        write_json(output_dir / 'error.json', {'status': 'failed', 'error_type': type(error).__name__,
                                              'error': str(error), 'traceback': traceback.format_exc(),
                                              'command': [sys.executable, *sys.argv]})
        raise


if __name__ == '__main__':
    main()
