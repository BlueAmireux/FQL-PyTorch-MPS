"""Atomic complete PyTorch checkpoints; JSON marker is the commit boundary."""
import os
from pathlib import Path
import torch
from .agent import FQLAgent
from .runtime import sha256, source_hashes, synchronize, write_json

SCHEMA = 1


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def capture_rng(device):
    states = {'cpu': torch.get_rng_state().clone(), 'mps': None}
    if torch.device(device).type == 'mps':
        states['mps'] = torch.mps.get_rng_state().clone()
    return states


def restore_rng(states, device):
    torch.set_rng_state(states['cpu'].cpu())
    if torch.device(device).type == 'mps':
        if states['mps'] is None:
            raise ValueError('Cannot strictly resume CPU randomness on MPS.')
        torch.mps.set_rng_state(states['mps'].cpu())


def save_checkpoint(run_dir, epoch, agent, config, sampler_state, probe_observations):
    run_dir = Path(run_dir)
    path = run_dir / f'checkpoint_{epoch}.pt'
    marker = run_dir / f'checkpoint_{epoch}.json'
    if path.exists() or marker.exists():
        raise FileExistsError(f'Existing checkpoint is immutable: {path}')
    if agent.updates != epoch:
        raise ValueError(f'Agent updates {agent.updates} != checkpoint epoch {epoch}')
    synchronize(agent.device)
    generator = torch.Generator(device='cpu').manual_seed(55)
    observations = torch.as_tensor(probe_observations, dtype=torch.float32)[:4].cpu()
    noises = torch.randn((len(observations), 5), generator=generator)
    actions = agent.sample_actions(observations.to(agent.device), noises.to(agent.device)).cpu()
    payload = {
        'schema_version': SCHEMA, 'completed_updates': epoch,
        'training_device': agent.device.type, 'config': config,
        'model': cpu_tree(agent.state_dict()), 'optimizer': cpu_tree(agent.optimizer.state_dict()),
        'rng': capture_rng(agent.device), 'sampler': cpu_tree(sampler_state),
        'source_sha256': source_hashes(),
        'reload_probe': {'observations': observations, 'noises': noises, 'actions': actions},
    }
    temp = path.with_suffix('.pt.tmp')
    with temp.open('xb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    # Never consider a .pt file without this final commit record resumable.
    write_json(marker, {'schema_version': SCHEMA, 'completed_updates': epoch,
                       'checkpoint_file': path.name, 'sha256': sha256(path),
                       'bytes': path.stat().st_size, 'training_device': agent.device.type,
                       'source_sha256': payload['source_sha256']}, exclusive=True)
    return marker


def load_checkpoint(run_dir, epoch, device='cpu', restore_rng=False):
    import json
    run_dir = Path(run_dir)
    marker_path = run_dir / f'checkpoint_{epoch}.json'
    path = run_dir / f'checkpoint_{epoch}.pt'
    marker = json.loads(marker_path.read_text())
    if marker['completed_updates'] != epoch or marker['checkpoint_file'] != path.name:
        raise ValueError('Checkpoint commit marker identity mismatch.')
    if sha256(path) != marker['sha256']:
        raise ValueError('Checkpoint SHA mismatch; refuse partial/corrupt checkpoint.')
    if marker['source_sha256'] != source_hashes():
        raise ValueError('Core source changed since checkpoint; explicit migration needed.')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['schema_version'] != SCHEMA or payload['completed_updates'] != epoch:
        raise ValueError('Unexpected checkpoint schema/count.')
    config = payload['config']
    agent = FQLAgent(config['agent'], device=device, seed=config['seed'])
    agent.load_state_dict(payload['model'], strict=True)
    agent.optimizer.load_state_dict(payload['optimizer'])
    agent.updates = epoch
    if restore_rng:
        if torch.device(device).type != payload['training_device']:
            raise ValueError('Strict training resume requires the original device type.')
        globals()['restore_rng'](payload['rng'], device)
    return agent, config, payload
