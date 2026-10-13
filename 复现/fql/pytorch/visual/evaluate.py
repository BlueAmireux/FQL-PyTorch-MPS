#!/usr/bin/env python3
"""Independent CPU closed-loop evaluation of a visual student or state teacher.

The student receives RGB, live robot proprioception (19), and Gaussian noise (5).
True cube state is saved for auditing, but is never passed to the student.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
ENV_NAME = 'visual-cube-single-singletask-task2-v0'
PROBE_STEPS = (0, 50, 100, 150)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--step', type=int)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--env-seed-start', type=int, default=40000)
    parser.add_argument('--action-seed-start', type=int, default=41000)
    parser.add_argument('--policy', choices=('student', 'teacher'), default='student')
    parser.add_argument('--image-mode', choices=('normal', 'zero'), default='normal')
    args = parser.parse_args()
    if args.policy == 'student' and (args.run is None or args.step is None):
        parser.error('--run and --step are required for the student.')
    if args.step is not None and args.step < 0:
        parser.error('--step must be nonnegative.')
    if args.episodes < 1 or min(args.env_seed_start, args.action_seed_start) < 0:
        parser.error('Episodes must be positive and seeds nonnegative.')
    if max(args.env_seed_start, args.action_seed_start) + args.episodes > 2**32:
        parser.error('Every environment/action seed must fit uint32.')
    if args.policy == 'teacher' and args.image_mode != 'normal':
        parser.error('The state teacher has no image input; use --image-mode normal.')
    return args


def checked_image(image, np):
    image = np.asarray(image)
    if image.shape != (64, 64, 3) or image.dtype != np.uint8:
        raise ValueError('Expected official RGB uint8 observation [64,64,3].')
    return image


def checked_state(state, np):
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (28,) or not np.isfinite(state).all():
        raise ValueError('Expected a finite live state reference [28].')
    if not (-1e-6 <= state[17] <= 3 + 1e-6 and -1e-6 <= state[18] <= 1 + 1e-6):
        raise ValueError('Live gripper opening/contact are outside official scaling.')
    for section in (state[15:17], state[22:26], state[26:28]):
        if not np.isclose(np.linalg.norm(section), 1, atol=1e-3, rtol=0):
            raise ValueError('Live yaw/quaternion reference is not normalized.')
    return state


def official_success(info, np):
    value = np.asarray(info['success'])
    if value.shape != () or not np.isfinite(value) or float(value) not in (0, 1):
        raise ValueError("Expected scalar binary official info['success'].")
    return bool(value)


def wilson_interval(successes, count):
    z = 1.959963984540054
    fraction = successes / count
    denominator = 1 + z * z / count
    center = (fraction + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def check_reload(model, payload, requested_step, torch, np):
    if int(payload['step']) != requested_step:
        raise ValueError('Loaded student step differs from the requested checkpoint.')
    probe = payload['probe']
    images = torch.as_tensor(probe['images'], device='cpu')
    proprio = torch.as_tensor(probe['proprio'], dtype=torch.float32, device='cpu')
    noises = torch.as_tensor(probe['noises'], dtype=torch.float32, device='cpu')
    expected = torch.as_tensor(probe['actions'], dtype=torch.float32, device='cpu').numpy()
    batch_size = images.shape[0] if images.ndim == 4 else 0
    if (batch_size < 1 or tuple(images.shape[1:]) != (3, 64, 64)
            or images.dtype != torch.uint8 or tuple(proprio.shape) != (batch_size, 19)
            or tuple(noises.shape) != (batch_size, 5) or expected.shape != (batch_size, 5)):
        raise ValueError('Invalid student reload probe shapes or image dtype.')
    if not (torch.isfinite(proprio).all() and torch.isfinite(noises).all()
            and np.isfinite(expected).all()):
        raise ValueError('Student reload probe contains nonfinite inputs or actions.')
    with torch.inference_mode():
        actual = model(images, proprio, noises).detach().cpu().numpy()
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError('Student reload produced invalid actions.')
    difference = float(np.max(np.abs(actual - expected)))
    if difference > 1e-4:
        raise ValueError(f'CPU student reload probe failed: max absolute difference {difference}.')
    return {'passed': True, 'max_absolute_action_difference': difference,
            'absolute_tolerance': 1e-4, 'probe_count': batch_size}


def evaluate(args, output):
    # Keep --help independent of the ML/environment installation.
    sys.path.insert(0, str(HERE.parent))
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from visual.common import (load_student_checkpoint, load_teacher, make_env,
                               sha256, state_observation, write_json)

    torch.set_num_threads(4)
    started = time.perf_counter()
    if args.policy == 'student':
        model, payload = load_student_checkpoint(args.run.resolve(), args.step, device='cpu', resume=False)
        model.eval()
        reload_check = check_reload(model, payload, args.step, torch, np)
        teacher_identity = payload.get('teacher')
    else:
        model, teacher_identity = load_teacher(device='cpu')
        model.eval()
        reload_check = {'passed': True, 'note': 'Frozen state teacher loaded by visual.common.'}
    if any(parameter.device.type != 'cpu' for parameter in model.parameters()):
        raise ValueError('This evaluator requires all policy parameters on CPU.')

    env = make_env()
    writer = None
    rows, terminal_states, first_images = [], [], []
    offsets = [0]
    replay = {name: [] for name in ('states', 'noises', 'actions', 'rewards',
                                  'successes', 'terminated', 'truncated')}
    image_probes, probe_action_indices = [], []
    inference_seconds = []
    try:
        if env.spec.id != ENV_NAME or env.spec.max_episode_steps != 200:
            raise ValueError('Expected the official visual cube task 2 with a 200-step horizon.')
        if (env.observation_space.shape != (64, 64, 3) or env.action_space.shape != (5,)
                or not np.all(env.action_space.low == -1) or not np.all(env.action_space.high == 1)):
            raise ValueError('Unexpected official image/action spaces.')
        if getattr(env.unwrapped, '_reward_task_id', None) != 2:
            raise ValueError('Expected fixed single-task reward ID 2.')

        columns = ['episode', 'env_seed', 'action_seed', 'return', 'length', 'success',
                   'ever_success', 'terminated', 'truncated', 'episode_seconds']
        with (output / 'episodes.csv').open('x', newline='', encoding='utf-8') as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=columns)
            csv_writer.writeheader()
            for episode in range(args.episodes):
                episode_started = time.perf_counter()
                env_seed = args.env_seed_start + episode
                action_seed = args.action_seed_start + episode
                image, _ = env.reset(seed=env_seed)
                image = checked_image(image, np)
                # Live dynamics supply contact; no state restoration occurs in this loop.
                state = checked_state(state_observation(env), np)
                generator = torch.Generator(device='cpu').manual_seed(action_seed)
                if episode == 0:
                    writer = imageio.get_writer(output / 'episode_000.mp4', fps=20, macro_block_size=1)
                    first_images.append(image.copy())
                    writer.append_data(image)
                episode_return, ever_success, done, length = 0.0, False, False, 0
                while not done:
                    noise = torch.randn((1, 5), generator=generator, dtype=torch.float32, device='cpu')
                    chw = np.ascontiguousarray(image.transpose(2, 0, 1))
                    if args.image_mode == 'zero':
                        chw = np.zeros_like(chw)
                    inference_started = time.perf_counter()
                    with torch.inference_mode():
                        if args.policy == 'student':
                            action_tensor = model(torch.from_numpy(chw).unsqueeze(0),
                                                  torch.from_numpy(state[:19].copy()).unsqueeze(0), noise)
                        else:
                            action_tensor = model(torch.from_numpy(state).unsqueeze(0), noise).clamp(-1, 1)
                    action = action_tensor.detach().cpu().numpy()
                    inference_seconds.append(time.perf_counter() - inference_started)
                    if action.shape != (1, 5) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
                        raise ValueError('Policy returned an invalid normalized action.')
                    action = action[0].astype(np.float32, copy=False)
                    if length in PROBE_STEPS:
                        image_probes.append(chw.copy())
                        probe_action_indices.append(len(replay['actions']))
                    next_image, reward, terminated, truncated, info = env.step(action)
                    next_image = checked_image(next_image, np)
                    next_state = checked_state(state_observation(env), np)
                    success = official_success(info, np)
                    if not np.isfinite(reward):
                        raise ValueError('Environment returned a nonfinite reward.')
                    for name, value in (('states', state.copy()), ('noises', noise[0].numpy().copy()),
                                        ('actions', action.copy()), ('rewards', float(reward)),
                                        ('successes', success), ('terminated', bool(terminated)),
                                        ('truncated', bool(truncated))):
                        replay[name].append(value)
                    episode_return += float(reward)
                    ever_success = ever_success or success
                    length += 1
                    image, state = next_image, next_state
                    done = bool(terminated or truncated)
                    if episode == 0:
                        first_images.append(image.copy())
                        writer.append_data(image)
                    if length >= 200 and not done:
                        raise RuntimeError('Official horizon reached without termination/truncation.')
                terminal_states.append(state.copy())
                offsets.append(len(replay['actions']))
                row = {'episode': episode, 'env_seed': env_seed, 'action_seed': action_seed,
                       'return': episode_return, 'length': length, 'success': int(success),
                       'ever_success': int(ever_success), 'terminated': bool(terminated),
                       'truncated': bool(truncated), 'episode_seconds': time.perf_counter() - episode_started}
                rows.append(row)
                csv_writer.writerow(row)
                stream.flush()
                print(json.dumps(row), flush=True)
                if episode == 0:
                    writer.close()
                    writer = None

        np.savez_compressed(output / 'replay.npz',
                            **{name: np.asarray(values) for name, values in replay.items()},
                            episode_offsets=np.asarray(offsets, dtype=np.int64),
                            terminal_states=np.asarray(terminal_states, dtype=np.float32),
                            env_seeds=np.asarray([row['env_seed'] for row in rows], dtype=np.uint32),
                            action_seeds=np.asarray([row['action_seed'] for row in rows], dtype=np.uint32),
                            episode_0_images=np.asarray(first_images, dtype=np.uint8),
                            image_probes=np.asarray(image_probes, dtype=np.uint8),
                            probe_action_indices=np.asarray(probe_action_indices, dtype=np.int64))
        successes = sum(row['success'] for row in rows)
        sources = {path.name: sha256(path) for path in (Path(__file__), HERE / 'common.py', HERE / 'student.py')}
        summary = {
            'status': 'completed', 'scope': 'Single-student visual distillation feasibility; not original-paper visual FQL reproduction.',
            'policy': args.policy, 'image_mode': args.image_mode, 'device': 'cpu',
            'env_name': ENV_NAME, 'task_id': 2, 'horizon': 200,
            'run': str(args.run.resolve()) if args.run is not None else None,
            'checkpoint_step': args.step if args.policy == 'student' else None,
            'checkpoint_sha256': sha256(args.run / f'checkpoint_{args.step}.pt') if args.policy == 'student' else None,
            'training_source_sha256': payload['source_sha256'] if args.policy == 'student' else None,
            'teacher': teacher_identity, 'reload': reload_check,
            'episodes': len(rows), 'successes': successes, 'success_rate': successes / len(rows),
            'success_rate_wilson_95_percent_ci': wilson_interval(successes, len(rows)),
            'success_definition': "Final official info['success']; ever_success is reported separately.",
            'mean_return': float(np.mean([row['return'] for row in rows])),
            'mean_length': float(np.mean([row['length'] for row in rows])),
            'evaluation_environment_steps': len(replay['actions']), 'optimizer_updates': 0,
            'offline_dataset_loaded': False, 'env_seeds': [row['env_seed'] for row in rows],
            'action_seeds': [row['action_seed'] for row in rows],
            'noise_rule': 'Independent CPU torch.Generator(action_seed), float32 randn(1,5) each step.',
            'student_input': 'RGB CHW uint8, live state[:19] robot proprioception, noise5; true cube state excluded.',
            'state_reference': 'Live compute_ob_info assembled into 28 values; includes true cube only for teacher/audit, contact from live physics.',
            'closed_loop_state_restoration': False,
            'replay_layout': 'Concatenated pre-action states/noises/actions; episode_offsets index transitions; terminal_states stores final states.',
            'image_storage': 'Episode 0 raw HWC images include reset/final; CHW policy-input probes at steps 0,50,100,150 per episode, indexed by probe_action_indices.',
            'recomputable_actions': 'Teacher: all; student normal: episode 0 plus saved probes; student zero: all using zero RGB.',
            'video': {'path': 'episode_000.mp4', 'selection': 'Episode 0 regardless of success',
                      'resolution': [64, 64], 'fps': 20, 'content': 'Live RGB observations, including in zero-input evaluation.'},
            'mean_inference_ms': float(np.mean(inference_seconds) * 1000),
            'evaluation_seconds': time.perf_counter() - started,
            'command': [sys.executable, *sys.argv], 'torch_version': torch.__version__,
            'source_sha256': sources, 'replay_sha256': sha256(output / 'replay.npz'),
            'limitations': ['One visual student training seed; fixed cases are not multi-seed visual reproduction.',
                            'Hybrid RGB plus robot proprioception differs from a pixel-only policy.',
                            'Final test cases must remain excluded from training and checkpoint selection.'],
        }
        write_json(output / 'summary.json', summary)
        print(json.dumps({'successes': successes, 'episodes': len(rows), 'output': str(output)}), flush=True)
    finally:
        if writer is not None:
            writer.close()
        env.close()


def main():
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        evaluate(args, output)
    except Exception as error:
        with (output / 'error.json').open('x', encoding='utf-8') as stream:
            json.dump({'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
                       'traceback': traceback.format_exc(), 'command': [sys.executable, *sys.argv]},
                      stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        raise


if __name__ == '__main__':
    main()
