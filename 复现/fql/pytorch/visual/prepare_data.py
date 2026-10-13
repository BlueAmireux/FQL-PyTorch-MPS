"""Render a small trajectory-separated subset of the existing official state data."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from visual.common import ROOT, make_env, state_observation, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sampler = np.random.default_rng(20261013)
    env = make_env()
    env.reset(seed=39100)
    records = {}
    started = time.perf_counter()
    try:
        for split, count in [('train', 50), ('val', 10)]:
            source = ROOT / 'DataSet/fql' / ('cube-single-play-v0.npz' if split == 'train'
                                          else 'cube-single-play-v0-val.npz')
            with np.load(source) as raw:
                ends = np.flatnonzero(raw['terminals'])
                starts = np.r_[0, ends[:-1] + 1]
                if not np.all(ends - starts == 1000):
                    raise ValueError('Unexpected trajectory layout.')
                trajectories = np.sort(sampler.choice(len(starts), count, replace=False))
                times = np.linspace(0, 999, 100, dtype=np.int64)
                indices = (starts[trajectories, None] + times).reshape(-1)
                states = raw['observations'][indices].copy()
                qpos, qvel = raw['qpos'][indices], raw['qvel'][indices]
                images = np.empty((len(indices), 3, 64, 64), dtype=np.uint8)
                geometry_error = 0.
                effector_error = 0.
                yaw_error, yaw_changed = 0., 0
                render_started = time.perf_counter()
                # Cube quaternion is geometry; yaw can be singular after float32 storage.
                stable_dims = np.r_[0:12, 17:18, 19:26]
                for i, (position, velocity) in enumerate(zip(qpos, qvel)):
                    env.unwrapped.set_state(position, velocity)
                    images[i] = env.unwrapped.compute_observation().transpose(2, 0, 1)
                    restored = state_observation(env)
                    geometry_error = max(geometry_error, float(np.max(np.abs(restored[stable_dims] - states[i, stable_dims]))))
                    effector_error = max(effector_error, float(np.max(np.abs(restored[12:15] - states[i, 12:15])) / 10))
                    yaw_difference = float(np.max(np.abs(restored[26:28] - states[i, 26:28])))
                    yaw_error = max(yaw_error, yaw_difference)
                    yaw_changed += int(yaw_difference > 1e-5)
                seconds = time.perf_counter() - render_started
                if geometry_error > 1e-5 or not images.any(axis=(1, 2, 3)).all():
                    raise ValueError('Geometry mismatch or black image.')
                path = args.output / f'{split}.npz'
                np.savez_compressed(path, images=images, states=states, proprio=states[:, :19],
                                    raw_indices=indices, trajectories=np.repeat(trajectories, 100),
                                    times=np.tile(times, count))
                records[split] = {'source': str(source), 'source_sha256': sha256(source),
                    'path': str(path.resolve()), 'sha256': sha256(path), 'bytes': path.stat().st_size,
                    'images': len(images), 'trajectory_ids': trajectories.tolist(),
                    'render_seconds': seconds, 'images_per_second': len(images) / seconds,
                    'qpos_derived_state_max_abs_error': geometry_error,
                    'cached_effector_max_abs_axis_error_m': effector_error,
                    'derived_cube_yaw_max_abs_error': yaw_error,
                    'derived_cube_yaw_changed_frames': yaw_changed}
                print(split, records[split], flush=True)
    finally:
        env.close()
    write_json(args.output / 'manifest.json', {'splits': records, 'selection_seed': 20261013,
        'frames_per_trajectory': 100, 'frame_rule': 'linspace(0,999,100); terminal row 1000 excluded',
        'environment': 'official visual task2, front_pixels, 64x64, visualize_info=False',
        'source_sha256': sha256(Path(__file__)), 'total_seconds': time.perf_counter() - started,
        'proprio_source': 'unaltered original observations[:19]; no contact reconstruction',
        'teacher_state_source': 'unaltered original observations[:28]',
        'limitation': 're-rendered state dataset, not official visual data; effector caches differ from mj_forward, cube yaw near Euler singularity may not survive float32 qpos storage; raw labels retained'}, exclusive=True)


if __name__ == '__main__':
    main()
