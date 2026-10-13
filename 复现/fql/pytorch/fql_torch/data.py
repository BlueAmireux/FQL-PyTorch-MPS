"""Official OGBench relabeling and packed in-memory uniform transition sampling."""
from pathlib import Path
import numpy as np
import torch
from .runtime import sha256

TASK = 'cube-single-play-singletask-v0'
FILES = {
    'cube-single-play-v0.npz': '80f3b6fd27f4f9d9e9eb6f0d07d6951559012f45b1e15ea4046ef8ecd8d3684e',
    'cube-single-play-v0-val.npz': '96d07401bdebdc3f0ea6d56ed1333863e0962f441483adc2c43b83105046eb00',
}
FIELDS = ('observations', 'actions', 'rewards', 'masks', 'next_observations')
SLICES = {'observations': slice(0, 28), 'actions': slice(28, 33),
          'rewards': slice(33, 34), 'masks': slice(34, 35), 'next_observations': slice(35, 63)}


def load_official(data_dir):
    import ogbench
    data_dir = Path(data_dir).resolve()
    manifest = {}
    for name, expected in FILES.items():
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f'Existing official dataset required; this runner never downloads: {path}')
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f'Dataset SHA mismatch: {path}')
        manifest[name] = {'path': str(path), 'sha256': actual, 'bytes': path.stat().st_size}
    env, train, val = ogbench.make_env_and_datasets(TASK, dataset_dir=str(data_dir))
    try:
        if env.observation_space.shape != (28,) or env.action_space.shape != (5,):
            raise ValueError('Unexpected task observation/action shapes.')
        for dataset, expected_count in ((train, 1000000), (val, 100000)):
            if len(dataset['actions']) != expected_count:
                raise ValueError('Official valid transition count changed.')
            dataset['actions'] = np.clip(dataset['actions'], -1 + 1e-5, 1 - 1e-5)
            for key in FIELDS:
                if dataset[key].dtype != np.float32 or not np.isfinite(dataset[key]).all():
                    raise ValueError(f'Nonfloat32/nonfinite official data: {key}')
            if not np.array_equal(dataset['masks'], -dataset['rewards']):
                raise ValueError('Task2 mask must be 1-success, reward=success-1.')
        manifest.update(task=TASK, train_transitions=len(train['actions']),
                        validation_transitions=len(val['actions']), action_clip_eps=1e-5,
                        mask_zero_count=int(np.count_nonzero(train['masks'] == 0)),
                        terminal_with_mask_one=int(np.count_nonzero((train['terminals'] == 1) & (train['masks'] == 1))))
    finally:
        env.close()
    return train, val, manifest


class ResidentDataset:
    """One index_select per batch; five fields share a packed float32 storage."""
    def __init__(self, arrays, device='mps', placement='device', seed=0):
        self.device = torch.device(device)
        self.placement = placement
        packed = np.concatenate([arrays[k].reshape(len(arrays[k]), -1) for k in FIELDS], axis=1)
        self.storage = torch.from_numpy(packed)
        if placement == 'device':
            self.storage = self.storage.to(self.device)
        elif placement != 'cpu':
            raise ValueError('placement must be device or cpu.')
        self.size = len(packed)
        self.sampler = np.random.RandomState(seed)

    def sample(self, batch_size, return_indices=False):
        indices = self.sampler.randint(self.size, size=batch_size)
        idx = torch.from_numpy(indices).to(self.storage.device)
        selected = self.storage.index_select(0, idx)
        if selected.device != self.device:
            selected = selected.to(self.device)
        batch = {k: selected[:, s] for k, s in SLICES.items()}
        batch['rewards'] = batch['rewards'].squeeze(-1)
        batch['masks'] = batch['masks'].squeeze(-1)
        return (batch, indices) if return_indices else batch

    def sampler_state(self):
        name, keys, pos, has_gauss, cached = self.sampler.get_state()
        return {'name': name, 'keys': torch.from_numpy(keys.copy().astype(np.int64)),
                'pos': pos, 'has_gauss': has_gauss, 'cached_gaussian': cached}

    def restore_sampler(self, state):
        self.sampler.set_state((state['name'], state['keys'].cpu().numpy().astype(np.uint32),
                                state['pos'], state['has_gauss'], state['cached_gaussian']))
