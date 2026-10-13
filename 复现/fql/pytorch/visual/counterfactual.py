"""Nine static cube positions with fixed robot proprioception and policy noise.

This CPU probe is not a rollout, task score, or training run. Baseline encoder
latents are not pose estimates; pose metrics require explicit XYZ supervision.
"""
import argparse
import importlib.metadata
import inspect
import itertools
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def probe(args, output):
    sys.path.insert(0, str(HERE.parent))
    import numpy as np
    import torch
    from visual.common import (load_student_checkpoint, load_teacher, make_env,
                               sha256, state_observation, write_json)
    from visual.student import VisualStudent

    torch.set_num_threads(4)
    started = time.perf_counter()
    run = args.run.resolve()
    model, payload = load_student_checkpoint(run, args.step, 'cpu', resume=False)
    if not isinstance(model, VisualStudent) or any(p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('Expected the visual student on CPU.')
    teacher, teacher_identity = load_teacher('cpu')
    if payload['teacher']['sha256'] != teacher_identity['sha256']:
        raise ValueError('Student and reference teacher identities differ.')
    config = payload['config']
    if json.loads((run / 'config.json').read_text()) != config:
        raise ValueError('Run config differs from the checkpoint config.')
    sources = {str(path): sha256(path) for path in
               (Path(__file__), HERE / 'common.py', HERE / 'student.py', HERE / 'train.py',
                Path(inspect.getfile(type(model.actor))))}
    for name, digest in config.get('additional_source_sha256', {}).items():
        path = HERE / name
        if sha256(path) != digest:
            raise ValueError(f'Additional training source changed: {name}')
        sources[str(path)] = digest
    grounded = float(config.get('xyz_loss_weight', 0)) > 0

    env = make_env()
    try:
        env.reset(seed=39000)
        base = env.unwrapped
        # Refresh derived robot kinematics once before freezing the proprio input.
        base.set_state(base.data.qpos.copy(), base.data.qvel.copy())
        base_state = state_observation(env)
        fixed_proprio = base_state[:19].copy()
        fixed_qpos, fixed_qvel = base.data.qpos.copy(), base.data.qvel.copy()
        joint = base.model.joint('object_joint_0')
        qa, va = int(joint.qposadr[0]), int(joint.dofadr[0])
        assert qa + 7 == base.model.nq and va + 6 == base.model.nv
        for cls in type(base).__mro__[:3]:
            path = Path(inspect.getfile(cls))
            sources[str(path)] = sha256(path)
        images, states, actual_proprio = [], [], []
        xyz = np.asarray([(x, y, .02) for x, y in
                          itertools.product((.32, .40, .52), (-.10, 0., .10))], dtype=np.float32)
        for position in xyz:
            qpos, qvel = fixed_qpos.copy(), fixed_qvel.copy()
            qpos[qa:qa + 7] = [*position, 1., 0., 0., 0.]
            qvel[va:va + 6] = 0
            base.set_state(qpos, qvel)
            if not (np.array_equal(base.data.qpos[:qa], fixed_qpos[:qa])
                    and np.array_equal(base.data.qvel[:va], fixed_qvel[:va])):
                raise ValueError('Robot joint state changed across cube cases.')
            state = state_observation(env)
            image = base.compute_observation().copy()
            if image.shape != (64, 64, 3) or image.dtype != np.uint8 or not np.any(image):
                raise ValueError('Invalid official RGB image.')
            images.append(image)
            states.append(state)
            actual_proprio.append(state[:19])
        images, states, actual_proprio = map(np.asarray, (images, states, actual_proprio))
        differences = np.abs(actual_proprio - fixed_proprio)
        if float(differences[:, :18].max()) > 1e-5:
            raise ValueError('Non-contact robot proprioception changed across cases.')
    finally:
        env.close()

    image_tensor = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2)))
    proprio = torch.from_numpy(fixed_proprio).unsqueeze(0).expand(9, -1)
    noise = torch.randn((1, 5), generator=torch.Generator(device='cpu').manual_seed(614))
    noises = noise.expand(9, -1)
    model.eval()
    with torch.inference_mode():
        actions = model(image_tensor, proprio, noises).numpy()
        zero_actions = model(torch.zeros_like(image_tensor), proprio, noises).numpy()
        raw_actions = model.raw_actions(image_tensor, proprio, noises).numpy()
        teacher_raw = teacher(torch.from_numpy(states), noises).numpy()
        teacher_actions = np.clip(teacher_raw, -1, 1)
        latent = model.encoder(image_tensor.float() / 255.).numpy()
        zero_latent = model.encoder(torch.zeros_like(image_tensor, dtype=torch.float32)).numpy()
    center = np.array([.425, 0., 0.], dtype=np.float32)
    pred_xyz, zero_xyz = latent[:, :3] / 10 + center, zero_latent[:, :3] / 10 + center
    if not all(np.isfinite(v).all() for v in (actions, zero_actions, raw_actions, teacher_raw, latent, pred_xyz)):
        raise ValueError('Nonfinite counterfactual model output.')
    with (output / 'counterfactual.npz').open('xb') as stream:
        np.savez_compressed(stream, images=images, fixed_proprio=fixed_proprio, fixed_noise=noise.numpy()[0],
                            actions=actions, zero_actions=zero_actions, teacher_actions=teacher_actions,
                            raw_actions=raw_actions, teacher_raw_actions=teacher_raw,
                            true_xyz=xyz, pred_xyz=pred_xyz, zero_pred_xyz=zero_xyz,
                            encoder_latent=latent, teacher_states=states, actual_proprio=actual_proprio,
                            robot_qpos=fixed_qpos[:qa], robot_qvel=fixed_qvel[:va])
    record = {
        'status': 'completed', 'scope': 'Nine static counterfactuals; not nine tasks or closed-loop evaluation.',
        'device': 'cpu', 'torch_mps_used': False, 'training_updates': 0,
        'evaluation_episodes': 0, 'external_environment_steps': 0,
        'reset_internal_goal_stabilization_steps': 2,
        'env_name': env.spec.id, 'task_id': 2, 'camera': 'front_pixels',
        'env_seed': 39000, 'noise_seed': 614, 'cases': 9,
        'cube_grid_m': {'x': [.32, .40, .52], 'y': [-.10, 0., .10], 'z': .02},
        'cube_quaternion_wxyz': [1., 0., 0., 0.], 'cube_velocity': 'zero',
        'robot_joints_identical': True, 'student_inputs': 'RGB, fixed robot proprio19, fixed noise5 only',
        'proprio_max_abs_difference_by_case': differences.max(1).tolist(),
        'noncontact_proprio_max_abs_difference': float(differences[:, :18].max()),
        'base_contact': float(fixed_proprio[18]),
        'contact_difference_by_case': (actual_proprio[:, 18] - fixed_proprio[18]).tolist(),
        'contact_note': 'Base contact cache is preserved; no forces are reconstructed or cube labels injected into proprio.',
        'unique_images': len({image.tobytes() for image in images}),
        'image_mean_abs_change_from_first': np.abs(images.astype(np.float32) - images[0]).mean((1, 2, 3)).tolist(),
        'normal_action_teacher_clipped_mse': float(np.mean((actions - teacher_actions) ** 2)),
        'normal_action_teacher_raw_mse': float(np.mean((raw_actions - teacher_raw) ** 2)),
        'zero_action_teacher_clipped_mse': float(np.mean((zero_actions - teacher_actions) ** 2)),
        'normal_zero_action_mean_abs_change': float(np.mean(np.abs(actions - zero_actions))),
        'normal_zero_action_l2_change_by_case': np.linalg.norm(actions - zero_actions, axis=1).tolist(),
        'normal_action_std_by_dimension': actions.std(0).tolist(),
        'zero_action_max_abs_change_from_first': float(np.abs(zero_actions - zero_actions[0]).max()),
        'xyz_supervised': grounded, 'xyz_loss_weight': config.get('xyz_loss_weight', 0),
        'pose_metrics': None,
        'pose_note': 'Encoder outputs 0:3 are supervised centered XYZ only for grounded checkpoints; baseline latent decoding is not a pose estimate.',
        'student': {'run': str(run), 'step': args.step,
                    'checkpoint_sha256': sha256(run / f'checkpoint_{args.step}.pt'),
                    'marker_sha256': sha256(run / f'checkpoint_{args.step}.json'),
                    'config_sha256': sha256(run / 'config.json'),
                    'reload_max_abs_error': payload['reload_max_abs_error'],
                    'checkpoint_source_sha256': payload['source_sha256'],
                    'additional_training_source_sha256': config.get('additional_source_sha256', {})},
        'teacher': teacher_identity, 'source_sha256': sources,
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'ogbench', 'mujoco')},
        'artifact_sha256': {'counterfactual.npz': sha256(output / 'counterfactual.npz')},
        'seconds': time.perf_counter() - started,
    }
    if grounded:
        error = (pred_xyz - xyz) * 100
        record['pose_metrics'] = {'xyz_axis_rmse_cm': np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
                                  'xyz_euclidean_rmse_cm': float(np.sqrt(np.mean(np.sum(error ** 2, axis=1)))),
                                  'zero_image_xyz_euclidean_rmse_cm': float(np.sqrt(np.mean(np.sum(((zero_xyz - xyz) * 100) ** 2, axis=1))))}
    write_json(output / 'record.json', record, exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.step < 0:
        parser.error('--step must be nonnegative.')
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        probe(args, args.output)
    except Exception as error:
        with (args.output / 'error.json').open('x') as stream:
            json.dump({'status': 'failed', 'error': str(error), 'traceback': traceback.format_exc()}, stream, indent=2)
        raise


if __name__ == '__main__':
    main()
