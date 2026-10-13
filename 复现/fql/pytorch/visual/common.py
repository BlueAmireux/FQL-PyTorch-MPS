"""Small, separate helpers for the RGB/proprioception transfer pilot."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from fql_torch.checkpoint import cpu_tree, capture_rng, restore_rng, load_checkpoint
from fql_torch.runtime import sha256, write_json, synchronize
from visual.student import VisualStudent

ROOT = Path(__file__).resolve().parents[4]
TEACHER_RUN = ROOT / 'experiment/fql_torch_migration/random_mps1000'
TEACHER_STEP = 960000
TEACHER_SHA = 'f0ad91c2d08b82258e94d6de46a4446487ac5e215142091a3ec25e32388056af'
ENV_NAME = 'visual-cube-single-singletask-task2-v0'


def make_env():
    import gymnasium as gym
    import ogbench  # Register the official environments, without downloading data.
    env = gym.make(ENV_NAME, width=64, height=64, visualize_info=False,
                   disable_env_checker=True)
    if env.unwrapped._reward_task_id != 2 or env.spec.max_episode_steps != 200:
        raise ValueError('Unexpected task or horizon.')
    if env.unwrapped.control_timestep() != 0.05:
        raise ValueError('Expected 20 Hz control.')
    return env


def state_observation(env):
    """Exact official state scaling, using the live post-step information."""
    info = env.unwrapped.compute_ob_info()
    center = np.array([0.425, 0., 0.])
    pieces = [info['proprio/joint_pos'], info['proprio/joint_vel'],
              (info['proprio/effector_pos'] - center) * 10,
              np.cos(info['proprio/effector_yaw']), np.sin(info['proprio/effector_yaw']),
              info['proprio/gripper_opening'] * 3, info['proprio/gripper_contact'],
              (info['privileged/block_0_pos'] - center) * 10,
              info['privileged/block_0_quat'], np.cos(info['privileged/block_0_yaw']),
              np.sin(info['privileged/block_0_yaw'])]
    state = np.concatenate(pieces).astype(np.float32)
    if state.shape != (28,) or not np.isfinite(state).all():
        raise ValueError('Invalid live state.')
    return state


def load_teacher(device='cpu'):
    path = TEACHER_RUN / f'checkpoint_{TEACHER_STEP}.pt'
    if sha256(path) != TEACHER_SHA:
        raise ValueError('Teacher identity changed.')
    agent, _, payload = load_checkpoint(TEACHER_RUN, TEACHER_STEP, 'cpu', False)
    probe = payload['reload_probe']
    with torch.inference_mode():
        actual = agent.sample_actions(probe['observations'], noises=probe['noises'])
    error = (actual - probe['actions']).abs().max().item()
    if not np.isfinite(error) or error > 1e-4:
        raise ValueError('State teacher reload probe failed.')
    actor = agent.actor_onestep_flow.eval().requires_grad_(False).to(device)
    identity = {'run': str(TEACHER_RUN), 'step': TEACHER_STEP, 'sha256': TEACHER_SHA,
                'seed': 0, 'selection': 'prior state validation-selected checkpoint',
                'reload_max_abs_error': error}
    return actor, identity


def visual_hashes():
    here = Path(__file__).parent
    return {name: sha256(here / name) for name in ('student.py', 'common.py', 'train.py')}


def save_student_checkpoint(run, step, model, optimizer, config, sampler, teacher,
                            images, proprio, noises):
    run = Path(run)
    path = run / f'checkpoint_{step}.pt'
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError(path)
    device = next(model.parameters()).device
    synchronize(device)
    with torch.inference_mode():
        actions = model(images, proprio, noises)
    payload = {'schema': 'visual-distillation-v1', 'step': step, 'config': config,
               'device': device.type, 'teacher': teacher, 'source_sha256': visual_hashes(),
               'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()),
               'rng': capture_rng(device), 'sampler': sampler.bit_generator.state,
               'probe': cpu_tree({'images': images, 'proprio': proprio, 'noises': noises,
                                  'actions': actions})}
    temp = path.with_suffix('.pt.tmp')
    with temp.open('xb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    write_json(path.with_suffix('.json'), {'step': step, 'sha256': sha256(path),
               'bytes': path.stat().st_size, 'source_sha256': payload['source_sha256']}, exclusive=True)


def load_student_checkpoint(run, step, device='cpu', resume=False):
    path = Path(run) / f'checkpoint_{step}.pt'
    marker = json.loads(path.with_suffix('.json').read_text())
    if marker['step'] != step or sha256(path) != marker['sha256']:
        raise ValueError('Visual checkpoint identity failed.')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['schema'] != 'visual-distillation-v1' or payload['step'] != step:
        raise ValueError('Unexpected visual checkpoint schema.')
    if payload['source_sha256'] != visual_hashes():
        raise ValueError('Visual training source changed.')
    model = VisualStudent().to(device)
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    probe = payload['probe']
    with torch.inference_mode():
        actual = model(*(probe[key].to(device) for key in ('images', 'proprio', 'noises'))).cpu()
    error = (actual - probe['actions']).abs().max().item()
    if not np.isfinite(error) or error > 1e-4:
        raise ValueError('Visual reload action probe failed.')
    payload['reload_max_abs_error'] = error
    if resume:
        if payload['device'] != torch.device(device).type:
            raise ValueError('Training resume requires the same device.')
        restore_rng(payload['rng'], device)
    return model, payload
