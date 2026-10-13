#!/usr/bin/env python3
"""Independent fixed CPU/MuJoCo evaluation of the average5 deployment policy."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time
import traceback

from policy import ENV_NAME, ROOT, load
sys.path.insert(0, str(ROOT / '复现/fql/pytorch'))
from evaluate import success_from_info, wilson_interval

EPISODES = 100
ENV_SEED_BASE = 9000
ACTION_SEED_BASE = 10000


def save_trajectory(path, trajectory, env_seed, action_seed, np):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **{key: np.asarray(value) for key, value in trajectory.items()},
                            env_seed=env_seed, action_seed=action_seed)


def evaluate(args, output_dir):
    import numpy as np
    import torch
    import ogbench
    from fql_torch.runtime import configure_device, memory, runtime_record, sha256, write_json
    device = configure_device('cpu', 4)
    started = time.perf_counter()
    env = None
    try:
        policy, artifact, marker, reload_check = load(args.policy_dir, device='cpu')
        write_json(output_dir / 'reload_check.json', reload_check, exclusive=True)
        runtime = runtime_record(device)
        runtime.update(evaluation_script_sha256=sha256(Path(__file__)),
                       policy_sha256=marker['sha256'], training_seed=artifact['training_seed'],
                       completed_updates=artifact['completed_updates'], offline_dataset_loaded=False,
                       optimizer_updates=0, reload_passed=True)
        write_json(output_dir / 'runtime.json', runtime, exclusive=True)
        env = ogbench.make_env_and_datasets(ENV_NAME, env_only=True)
        if (env.unwrapped._reward_task_id != 2 or env.spec.max_episode_steps != 200
                or env.observation_space.shape != (28,) or env.action_space.shape != (5,)):
            raise ValueError('Task/horizon/spaces differ from the frozen original environment.')
        write_json(output_dir / 'environment.json', {'env_name': ENV_NAME, 'task_id': 2,
            'max_episode_steps': 200, 'observation_shape': [28], 'action_shape': [5],
            'episodes': EPISODES, 'env_seed_base': ENV_SEED_BASE, 'action_seed_base': ACTION_SEED_BASE,
            'success_definition': "Final official info['success']; ever_success recorded separately."}, exclusive=True)
        rows = []
        fields = ['episode', 'env_seed', 'action_seed', 'return', 'length', 'success',
                  'ever_success', 'terminated', 'truncated', 'trajectory']
        with (output_dir / 'episodes.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i in range(EPISODES):
                env_seed, action_seed = ENV_SEED_BASE + i, ACTION_SEED_BASE + i
                trajectory = {key: [] for key in ('observations', 'actions', 'noises', 'rewards',
                    'successes', 'terminated', 'truncated')}
                try:
                    obs, info = env.reset(seed=env_seed)
                    obs = np.asarray(obs, dtype=np.float32)
                    if obs.shape != (28,) or not np.isfinite(obs).all():
                        raise ValueError('Invalid reset observation.')
                    trajectory['observations'].append(obs.copy())
                    generator = torch.Generator(device='cpu').manual_seed(action_seed)
                    done = False
                    while not done:
                        if len(trajectory['actions']) >= 200:
                            raise RuntimeError('Incomplete episode at the 200-step horizon.')
                        z = torch.randn(5, generator=generator, dtype=torch.float32)
                        with torch.inference_mode():
                            action = policy(torch.from_numpy(obs), z).clamp(-1, 1).numpy().copy()
                        if action.shape != (5,) or not np.isfinite(action).all():
                            raise ValueError('Invalid policy action.')
                        obs, reward, terminated, truncated, info = env.step(action)
                        obs = np.asarray(obs, dtype=np.float32)
                        success = success_from_info(info, np)
                        if obs.shape != (28,) or not np.isfinite(obs).all() or not np.isfinite(reward):
                            raise ValueError('Invalid environment transition.')
                        for key, value in (('observations', obs.copy()), ('actions', action),
                            ('noises', z.numpy().copy()), ('rewards', float(reward)), ('successes', success),
                            ('terminated', bool(terminated)), ('truncated', bool(truncated))):
                            trajectory[key].append(value)
                        done = bool(terminated or truncated)
                except BaseException:
                    save_trajectory(output_dir / f'episode_{i:03d}_partial.npz', trajectory,
                                    env_seed, action_seed, np)
                    raise
                filename = f'episode_{i:03d}.npz'
                save_trajectory(output_dir / filename, trajectory, env_seed, action_seed, np)
                row = dict(episode=i, env_seed=env_seed, action_seed=action_seed,
                    **{'return': sum(trajectory['rewards'])}, length=len(trajectory['actions']),
                    success=int(success), ever_success=int(any(trajectory['successes'])),
                    terminated=bool(terminated), truncated=bool(truncated), trajectory=filename)
                writer.writerow(row)
                stream.flush()
                rows.append(row)
                print(json.dumps(row), flush=True)
        successes = sum(row['success'] for row in rows)
        write_json(output_dir / 'metrics.json', {'status': 'completed', 'policy_label': 'average5',
            'scope': 'Predeclared three-training-seed fixed-window deployment comparison.',
            'completed_updates': artifact['completed_updates'], 'max_source_updates': artifact['max_source_updates'],
            'training_seed': artifact['training_seed'], 'episodes': EPISODES, 'successes': successes,
            'env_seed_base': ENV_SEED_BASE, 'action_seed_base': ACTION_SEED_BASE,
            'success_rate': successes / EPISODES,
            'success_rate_wilson_95_percent_ci': wilson_interval(successes, EPISODES),
            'mean_return': float(np.mean([row['return'] for row in rows])),
            'mean_length': float(np.mean([row['length'] for row in rows])),
            'evaluation_environment_steps': sum(row['length'] for row in rows),
            'training_environment_steps': 0, 'optimizer_updates': 0,
            'max_reload_probe_error': reload_check['maximum_absolute_action_difference'],
            'policy_sha256': marker['sha256'], 'source_checkpoints': artifact['source_checkpoints'],
            'source_config_sha256': artifact['source_config_sha256'],
            'frozen_core_sha': artifact['frozen_core_sha'], 'evaluation_seconds': time.perf_counter() - started,
            'final_memory': memory(device)}, exclusive=True)
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        evaluate(args, output_dir)
    except BaseException as error:
        with (output_dir / 'error.json').open('x', encoding='utf-8') as stream:
            json.dump({'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
                       'traceback': traceback.format_exc(), 'command': [sys.executable, *sys.argv]},
                      stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        raise


if __name__ == '__main__':
    main()
