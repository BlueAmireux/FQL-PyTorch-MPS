"""CPU on-policy correction-data collection, separate from final evaluation.

Save every second pre-action RGB/state pair. Recorded teacher actions use the
observed noise for auditing; a new trainer must query the teacher with fresh noise.
"""
import argparse
import csv
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
STRIDE = 2


def reset_episode(env, seed, shifted):
    """Official reset, optionally followed by a checked cube-only initial shift."""
    import numpy as np
    from visual.common import state_observation
    image, _ = env.reset(seed=seed)
    base = env.unwrapped
    reset_state = state_observation(env)
    qpos, qvel = base.data.qpos.copy(), base.data.qvel.copy()
    goal, goal_quat, ctrl = base.data.mocap_pos.copy(), base.data.mocap_quat.copy(), base.data.ctrl.copy()
    joint = base.model.joint('object_joint_0')
    qa, va = int(joint.qposadr[0]), int(joint.dofadr[0])
    displacement = np.zeros(2)
    if shifted:
        generator = np.random.default_rng(seed + 1000000)
        displacement = generator.uniform([-.03, -.06], [.03, .06])
        changed_pos, changed_vel = qpos.copy(), qvel.copy()
        changed_pos[qa:qa + 2] += displacement
        changed_vel[va:va + 6] = 0
        base.set_state(changed_pos, changed_vel)
    if not (np.array_equal(base.data.qpos[:qa], qpos[:qa])
            and np.array_equal(base.data.qvel[:va], qvel[:va])
            and np.array_equal(base.data.mocap_pos, goal)
            and np.array_equal(base.data.mocap_quat, goal_quat)
            and np.array_equal(base.data.ctrl, ctrl)):
        raise ValueError('Initial shift changed robot state, goal, or control cache.')
    pad_ids = {base.model.body(f'ur5e/robotiq/{side}_pad').id for side in ('left', 'right')}
    for contact in base.data.contact[:base.data.ncon]:
        bodies = {int(base.model.geom_bodyid[int(contact.geom1)]), int(base.model.geom_bodyid[int(contact.geom2)])}
        if bodies & pad_ids and contact.efc_address >= 0:
            raise ValueError('Initial gripper has an active contact; zero-contact reset is invalid.')
    right_pad = base.data.body('ur5e/robotiq/right_pad')
    old_contact = float(np.clip(np.linalg.norm(right_pad.cfrc_ext) / 50, 0, 1))
    if shifted:
        # Artificial repositioning: old goal-stabilization ctrl is not a force snapshot.
        # With no active pad contacts, clear this diagnostic wrench only at this reset.
        right_pad.cfrc_ext[:] = 0
    state = state_observation(env)
    distance = float(np.linalg.norm(base.data.qpos[qa:qa + 3] - goal[base._cube_target_mocap_ids[0]]))
    if distance <= .06 or state[18] != 0:
        raise ValueError('Initial cube is too close to the goal or contact is nonzero.')
    if shifted:
        image = base.compute_observation().copy()
    audit = {'dx': float(displacement[0]), 'dy': float(displacement[1]),
             'initial_goal_distance_m': distance, 'reset_cached_contact_before_clear': old_contact,
             'initial_contact': float(state[18]), 'robot_goal_ctrl_unchanged': True,
             'contact_cache_cleared': bool(shifted),
             'initial_effector_observation_max_abs_change': float(np.abs(state[12:17] - reset_state[12:17]).max())}
    return image, state, audit


def collect(args, output):
    sys.path.insert(0, str(HERE.parent))
    import numpy as np
    import torch
    from visual.common import (load_student_checkpoint, load_teacher, make_env,
                               sha256, state_observation, write_json)
    from fql_torch.runtime import source_hashes

    torch.set_num_threads(4)
    started = time.perf_counter()
    env = None
    episode, length, collection_steps = -1, 0, 0
    records = {name: [] for name in ('images', 'states', 'proprio', 'episode_ids', 'behavior_id',
               'shifted', 'step_index', 'noises', 'executed_actions', 'teacher_raw_actions')}
    rows, env_seeds, action_seeds, terminal_states = [], [], [], []
    geometry = {'cube_state_qpos_max_abs_error_m': 0., 'cube_geometry_cache_qpos_max_axis_offset_m': 0.,
                'joint_state_qpos_max_abs_error': 0., 'joint_state_qvel_max_abs_error': 0.,
                'cube_quaternion_max_abs_error': 0.}
    manifest = {'status': 'running', 'purpose': 'New simulation training-data collection; not final evaluation.',
                'device': 'cpu', 'cpu_threads': 4, 'MPS_training_used': False,
                'sample_stride': STRIDE, 'student_episodes_requested': args.student_episodes,
                'split': args.split,
                'teacher_episodes_requested': args.teacher_episodes,
                'env_seed_start': args.env_seed_start, 'action_seed_start': args.action_seed_start,
                'command': [sys.executable, *sys.argv], 'training_updates': 0,
                'final_evaluation_episodes': 0, 'errors': []}
    try:
        run = args.student_run.resolve()
        student, payload = load_student_checkpoint(run, args.student_step, 'cpu', resume=False)
        teacher, teacher_identity = load_teacher('cpu')
        if payload['teacher']['sha256'] != teacher_identity['sha256']:
            raise ValueError('Student and teacher identities differ.')
        if json.loads((run / 'config.json').read_text()) != payload['config']:
            raise ValueError('Run and checkpoint configurations differ.')
        if any(p.device.type != 'cpu' for model in (student, teacher) for p in model.parameters()):
            raise ValueError('Both collection policies must be on CPU.')
        sources = {str(path): sha256(path) for path in
                   (Path(__file__), HERE / 'common.py', HERE / 'student.py', HERE / 'train.py')}
        for name, digest in payload['config'].get('additional_source_sha256', {}).items():
            path = HERE / name
            if sha256(path) != digest:
                raise ValueError(f'Additional training source changed: {name}')
            sources[str(path)] = digest
        manifest.update(student={'run': str(run), 'step': args.student_step,
                        'checkpoint_sha256': sha256(run / f'checkpoint_{args.student_step}.pt'),
                        'marker_sha256': sha256(run / f'checkpoint_{args.student_step}.json'),
                        'config_sha256': sha256(run / 'config.json'),
                        'reload_max_abs_error': payload['reload_max_abs_error'],
                        'checkpoint_source_sha256': payload['source_sha256'],
                        'additional_source_sha256': payload['config'].get('additional_source_sha256', {})},
                        teacher=teacher_identity, source_sha256=sources,
                        teacher_core_source_sha256=source_hashes(),
                        versions={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'ogbench', 'mujoco')})
        student.eval()
        env = make_env()
        base = env.unwrapped
        for cls in type(base).__mro__[:3]:
            path = Path(inspect.getfile(cls))
            sources[str(path)] = sha256(path)
        manifest.update(env_name=env.spec.id, task_id=2, horizon=200, camera='front_pixels',
                        initial_shift={'odd_episode_only': True, 'x_uniform_m': [-.03, .03],
                                       'y_uniform_m': [-.06, .06], 'seed_offset': 1000000,
                                       'cube_velocity': 'six components zero only on shifted resets'},
                        contact_policy='Standard reset is untouched. Only after shifted artificial initialization: verify no active pad contacts and clear right_pad external-wrench diagnostic cache. Keep ctrl unchanged; first real action overwrites it. Later contact is entirely official live step/rnePostConstraint.',
                        target_rule='teacher_raw_actions use observed noise only for audit; requery frozen teacher with fresh noise during new training',
                        limitations=['Teacher is a learned policy, not an oracle; failed trajectories are retained.',
                                     'Hybrid RGB/proprio19 differs from pixel-only control.',
                                     'Each call is an independent output split; no random train/val resplit or downloads.',
                                     'Live post-step geometry caches may lag integrated qpos by one physics step; no extra forward during rollout.'])
        columns = ['episode', 'behavior_id', 'behavior', 'shifted', 'env_seed', 'action_seed', 'length',
                   'samples', 'return', 'success', 'ever_success', 'terminated', 'truncated',
                   'dx', 'dy', 'initial_goal_distance_m', 'reset_cached_contact_before_clear',
                   'initial_contact', 'robot_goal_ctrl_unchanged', 'contact_cache_cleared',
                   'initial_effector_observation_max_abs_change', 'seconds']
        with (output / 'episodes.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for episode in range(args.student_episodes + args.teacher_episodes):
                episode_started = time.perf_counter()
                length = 0
                behavior = int(episode >= args.student_episodes)
                shifted = bool(episode % 2)
                es, az = args.env_seed_start + episode, args.action_seed_start + episode
                env_seeds.append(es)
                action_seeds.append(az)
                image, state, initial = reset_episode(env, es, shifted)
                joint = base.model.joint('object_joint_0')
                qa, body = int(joint.qposadr[0]), int(joint.bodyid[0])
                generator = torch.Generator(device='cpu').manual_seed(az)
                offset, length, total_return, ever_success = len(records['images']), 0, 0., False
                while True:
                    chw = np.ascontiguousarray(image.transpose(2, 0, 1))
                    if chw.shape != (3, 64, 64) or chw.dtype != np.uint8 or not np.any(chw):
                        raise ValueError('Invalid RGB before action.')
                    noise = torch.randn((1, 5), generator=generator, dtype=torch.float32, device='cpu')
                    with torch.inference_mode():
                        teacher_raw = teacher(torch.from_numpy(state).unsqueeze(0), noise)
                        action_tensor = teacher_raw.clamp(-1, 1) if behavior else student(
                            torch.from_numpy(chw).unsqueeze(0), torch.from_numpy(state[:19].copy()).unsqueeze(0), noise)
                    action, target = action_tensor.numpy()[0], teacher_raw.numpy()[0]
                    if (action.shape != (5,) or not np.isfinite(action).all()
                            or np.any(np.abs(action) > 1) or not np.isfinite(target).all()):
                        raise ValueError('Invalid executed or teacher audit action.')
                    if length % STRIDE == 0:
                        values = (chw.copy(), state.copy(), state[:19].copy(), episode, behavior, shifted,
                                  length, noise.numpy()[0].copy(), action.copy(), target.copy())
                        for name, value in zip(records, values):
                            records[name].append(value)
                        cube = base.data.qpos[qa:qa + 3]
                        geometry['cube_state_qpos_max_abs_error_m'] = max(geometry['cube_state_qpos_max_abs_error_m'], float(np.abs(state[19:22] / 10 + [.425, 0., 0.] - cube).max()))
                        geometry['cube_geometry_cache_qpos_max_axis_offset_m'] = max(geometry['cube_geometry_cache_qpos_max_axis_offset_m'], float(np.abs(base.data.xpos[body] - cube).max()))
                        geometry['joint_state_qpos_max_abs_error'] = max(geometry['joint_state_qpos_max_abs_error'], float(np.abs(state[:6] - base.data.qpos[base._arm_joint_ids]).max()))
                        geometry['joint_state_qvel_max_abs_error'] = max(geometry['joint_state_qvel_max_abs_error'], float(np.abs(state[6:12] - base.data.qvel[base._arm_joint_ids]).max()))
                        geometry['cube_quaternion_max_abs_error'] = max(geometry['cube_quaternion_max_abs_error'], float(np.abs(state[22:26] - base.data.qpos[qa + 3:qa + 7]).max()))
                    image, reward, terminated, truncated, info = env.step(action)
                    collection_steps += 1
                    length += 1
                    state = state_observation(env)
                    success = bool(info['success'])
                    if not np.isfinite(reward) or float(reward) != float(success) - 1:
                        raise ValueError('Official task2 reward/success mismatch.')
                    total_return += float(reward)
                    ever_success = ever_success or success
                    if terminated or truncated:
                        break
                    if length >= 200:
                        raise RuntimeError('Official 200-step horizon did not truncate.')
                row = dict(episode=episode, behavior_id=behavior, behavior='teacher' if behavior else 'student',
                           shifted=shifted, env_seed=es, action_seed=az, length=length,
                           samples=len(records['images']) - offset, **{'return': total_return}, success=int(success),
                           ever_success=int(ever_success), terminated=bool(terminated), truncated=bool(truncated),
                           seconds=time.perf_counter() - episode_started, **initial)
                rows.append(row)
                terminal_states.append(state.copy())
                writer.writerow(row)
                stream.flush()
                print(json.dumps(row), flush=True)
        manifest['status'] = 'completed'
    except Exception as error:
        manifest['status'] = 'failed'
        manifest['errors'].append({'type': type(error).__name__, 'message': str(error),
                                   'episode': episode, 'steps_in_current_episode': length,
                                   'traceback': traceback.format_exc()})
        raise
    finally:
        if env is not None:
            env.close()
        if records['images']:
            arrays = {name: np.asarray(values) for name, values in records.items()}
            for name in ('states', 'proprio', 'noises', 'executed_actions', 'teacher_raw_actions'):
                arrays[name] = arrays[name].astype(np.float32)
            for name in ('episode_ids', 'step_index'):
                arrays[name] = arrays[name].astype(np.int32)
            arrays['behavior_id'] = arrays['behavior_id'].astype(np.uint8)
            arrays['env_seeds'] = np.asarray(env_seeds, dtype=np.uint32)
            arrays['action_seeds'] = np.asarray(action_seeds, dtype=np.uint32)
            arrays['terminal_states'] = np.asarray(terminal_states, dtype=np.float32).reshape(-1, 28)
            with (output / 'data.npz').open('xb') as stream:
                np.savez_compressed(stream, **arrays)
            manifest.update(data_sha256=sha256(output / 'data.npz'), data_bytes=(output / 'data.npz').stat().st_size,
                            data_shapes={name: list(value.shape) for name, value in arrays.items()},
                            data_dtypes={name: str(value.dtype) for name, value in arrays.items()})
        manifest.update(samples=len(records['images']), completed_episodes=len(rows),
                        collection_environment_steps=collection_steps,
                        reset_internal_goal_stabilization_steps=2 * len(env_seeds), geometry=geometry,
                        student_successes=sum(row['success'] for row in rows if row['behavior_id'] == 0),
                        teacher_successes=sum(row['success'] for row in rows if row['behavior_id'] == 1),
                        seconds=time.perf_counter() - started)
        if (output / 'episodes.csv').exists():
            manifest['episodes_csv_sha256'] = sha256(output / 'episodes.csv')
        write_json(output / 'manifest.json', manifest, exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student-run', type=Path, required=True)
    parser.add_argument('--student-step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--student-episodes', type=int, required=True)
    parser.add_argument('--teacher-episodes', type=int, required=True)
    parser.add_argument('--env-seed-start', type=int, required=True)
    parser.add_argument('--action-seed-start', type=int, required=True)
    parser.add_argument('--split', choices=('train', 'val', 'probe'), required=True)
    args = parser.parse_args()
    total = args.student_episodes + args.teacher_episodes
    if (min(args.student_step, args.student_episodes, args.teacher_episodes,
            args.env_seed_start, args.action_seed_start) < 0 or total < 1
            or max(args.env_seed_start, args.action_seed_start) + total > 2**32):
        parser.error('Require nonnegative counts/step/seeds, at least one episode, and uint32 seeds.')
    args.output.mkdir(parents=True, exist_ok=False)
    collect(args, args.output.resolve())


if __name__ == '__main__':
    main()
