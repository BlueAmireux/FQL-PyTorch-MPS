"""Independent CPU evaluation of an inference-only averaged student policy."""
import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / '复现/fql/pytorch'
sys.path.insert(0, str(BASE))
from evaluate import ENV_NAME, success_from_info, wilson_interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--env-seed-base', type=int, required=True)
    parser.add_argument('--action-seed-base', type=int, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    import ogbench
    from fql_torch.networks import ActorVectorField
    from fql_torch.runtime import configure_device, memory, runtime_record, sha256, source_hashes, write_json
    device = configure_device('cpu', 4)
    started = time.perf_counter()
    env = None
    try:
        path = args.policy_dir / 'student_policy.pt'
        marker = json.loads((args.policy_dir / 'student_policy.json').read_text())
        if marker['sha256'] != sha256(path) or marker['bytes'] != path.stat().st_size:
            raise ValueError('Inference artifact SHA/size differs from its commit marker.')
        artifact = torch.load(path, map_location='cpu', weights_only=True)
        if artifact['schema'] != 'inference_only_student_average_v1' or not marker['inference_only'] or marker['resumable']:
            raise ValueError('Expected a committed, inference-only student average.')
        if artifact['frozen_core_sha'] != source_hashes():
            raise ValueError('Frozen baseline core source identity changed.')
        if artifact['generation_script_sha256'] != sha256(Path(__file__).with_name('build_policy.py')):
            raise ValueError('Policy generation script identity changed.')
        config = artifact['source_config']
        if config['env_name'] != ENV_NAME or config['seed'] != 0:
            raise ValueError('Expected the frozen seed0 task2 source chain.')
        c = config['agent']
        policy = ActorVectorField(28, 5, c['actor_hidden_dims'], c['actor_layer_norm'], with_time=False)
        policy.load_state_dict(artifact['student'], strict=True)
        policy.eval()
        probe = artifact['reload_probe']
        with torch.inference_mode():
            actual = policy(probe['observations'], probe['noises']).clamp(-1, 1)
        torch.testing.assert_close(actual, probe['actions'], atol=1e-6, rtol=0)
        reload_error = float((actual-probe['actions']).abs().max())
        write_json(args.output_dir / 'reload_check.json', {'passed': True, 'maximum_absolute_action_difference': reload_error,
                   'atol': 1e-6, 'rtol': 0, 'policy_sha256': marker['sha256'], 'inference_only': True}, exclusive=True)
        env = ogbench.make_env_and_datasets(ENV_NAME, env_only=True)
        if env.unwrapped._reward_task_id != 2 or env.spec.max_episode_steps != 200:
            raise ValueError('Task/horizon differs from the frozen original environment.')
        rows = []
        fields = ['episode', 'env_seed', 'action_seed', 'return', 'length', 'success', 'ever_success', 'terminated', 'truncated', 'trajectory']
        with (args.output_dir / 'episodes.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i in range(100):
                es, az = args.env_seed_base+i, args.action_seed_base+i
                obs, info = env.reset(seed=es)
                obs = np.asarray(obs, dtype=np.float32)
                generator = torch.Generator().manual_seed(az)
                trajectory = {'observations': [obs.copy()], 'actions': [], 'noises': [], 'rewards': [],
                              'successes': [], 'terminated': [], 'truncated': []}
                done = False
                try:
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
                        for key, value in (('observations', obs.copy()), ('actions', action), ('noises', z.numpy().copy()),
                                           ('rewards', float(reward)), ('successes', success),
                                           ('terminated', bool(terminated)), ('truncated', bool(truncated))):
                            trajectory[key].append(value)
                        done = bool(terminated or truncated)
                except BaseException:
                    np.savez_compressed(args.output_dir / f'episode_{i:03d}_partial.npz',
                        **{k: np.asarray(v) for k,v in trajectory.items()}, env_seed=es, action_seed=az)
                    raise
                filename = f'episode_{i:03d}.npz'
                np.savez_compressed(args.output_dir / filename, **{k: np.asarray(v) for k,v in trajectory.items()}, env_seed=es, action_seed=az)
                row = dict(episode=i, env_seed=es, action_seed=az, **{'return':sum(trajectory['rewards'])},
                    length=len(trajectory['actions']), success=int(success), ever_success=int(any(trajectory['successes'])),
                    terminated=bool(terminated), truncated=bool(truncated), trajectory=filename)
                writer.writerow(row)
                stream.flush()
                rows.append(row)
                print(json.dumps(row), flush=True)
        successes = sum(row['success'] for row in rows)
        write_json(args.output_dir / 'metrics.json', {'status': 'completed', 'policy_label': f'average{artifact["window"]}',
            'scope': 'Single-seed fixed-window inference-only policy development/confirmation; no algorithm training.',
            'max_source_updates': artifact['max_source_updates'], 'training_seed': 0, 'episodes': 100,
            'successes': successes, 'success_rate': successes/100, 'success_rate_wilson_95_percent_ci': wilson_interval(successes,100),
            'mean_return': float(np.mean([r['return'] for r in rows])), 'mean_length': float(np.mean([r['length'] for r in rows])),
            'evaluation_environment_steps': sum(row['length'] for row in rows), 'training_environment_steps': 0,
            'optimizer_updates': 0, 'max_reload_probe_error': reload_error, 'policy_sha256': marker['sha256'],
            'source_checkpoints': artifact['source_checkpoints'], 'evaluation_seconds': time.perf_counter()-started,
            'final_memory': memory(device)}, exclusive=True)
        write_json(args.output_dir / 'runtime.json', runtime_record(device), exclusive=True)
    except BaseException as error:
        write_json(args.output_dir / 'error.json', {'status': 'failed', 'error': str(error), 'type': type(error).__name__,
                   'traceback': traceback.format_exc()}, exclusive=True)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
