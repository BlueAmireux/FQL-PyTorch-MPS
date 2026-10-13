"""Frozen policy brightness diagnosis with raw-image evidence."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--step', type=int)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, required=True)
    parser.add_argument('--env-seed-start', type=int, required=True)
    parser.add_argument('--action-seed-start', type=int, required=True)
    parser.add_argument('--policy', choices=('student', 'teacher'), default='student')
    parser.add_argument('--image-mode', choices=('normal', 'zero'), default='normal')
    parser.add_argument('--suite', choices=('standard', 'shifted'), required=True)
    parser.add_argument("--brightness", type=float, choices=(1.0,0.7), required=True)
    args = parser.parse_args()
    if args.policy != "student" or args.image_mode != "normal":
        parser.error("Brightness diagnosis requires normal-image student inference.")
    if args.policy == 'student' and (args.run is None or args.step is None):
        parser.error('Student requires --run and --step.')
    if args.episodes < 1 or min(args.env_seed_start, args.action_seed_start) < 0:
        parser.error('Require positive episodes and nonnegative seeds.')
    if max(args.env_seed_start, args.action_seed_start) + args.episodes > 2**32:
        parser.error('Seeds must fit uint32.')
    if args.policy == 'teacher' and args.image_mode != 'normal':
        parser.error('Teacher has no image input.')

    import gymnasium as gym
    import torch
    from visual import common, evaluate as frozen_evaluator
    from visual.collect_corrections import reset_episode

    extra_sources = {}
    if args.policy == 'student':
        path = args.run / f'checkpoint_{args.step}.pt'
        payload = torch.load(path, map_location='cpu', weights_only=True)
        config = payload['config']
        if config != json.loads((args.run / 'config.json').read_text()):
            raise ValueError('Checkpoint and run configuration differ.')
        for name, digest in config.get('additional_source_sha256', {}).items():
            if common.sha256(HERE / name) != digest:
                raise ValueError(f'Additional training source changed: {name}')
            extra_sources[name] = digest
        del payload

    import numpy as np
    original_factory = common.make_env
    initializations = []

    class ShiftedReset(gym.Wrapper):
        def reset(self, *, seed=None, options=None):
            image, _, record = reset_episode(self.env, seed, shifted=True)
            initializations.append({'env_seed': seed, **record})
            return image, self.env.unwrapped.get_reset_info()

    class BrightnessObservation(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.episode = -1
            self.step_index = 0
            self.raw_episode0, self.raw_probes, self.probe_tags = [], [], []

        def observation(self, raw):
            if raw.dtype != np.uint8 or raw.shape != (64,64,3):
                raise ValueError('Unexpected raw RGB.')
            if self.episode == 0:
                self.raw_episode0.append(raw.copy())
            if self.step_index in frozen_evaluator.PROBE_STEPS:
                self.raw_probes.append(raw.transpose(2,0,1).copy())
                self.probe_tags.append((self.episode,self.step_index))
            return np.clip(np.rint(raw.astype(np.float32)*args.brightness),0,255).astype(np.uint8)

        def reset(self, *, seed=None, options=None):
            raw, info = self.env.reset(seed=seed,options=options)
            self.episode += 1
            self.step_index = 0
            return self.observation(raw), info

        def step(self, action):
            raw, reward, terminated, truncated, info = self.env.step(action)
            self.step_index += 1
            return self.observation(raw), reward, terminated, truncated, info

    wrapper = None
    def factory():
        nonlocal wrapper
        base = original_factory()
        if args.suite == 'shifted':
            base = ShiftedReset(base)
        wrapper = BrightnessObservation(base)
        return wrapper
    common.make_env = factory
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        frozen_evaluator.evaluate(args, output)
        summary_path = output / 'summary.json'
        summary = json.loads(summary_path.read_text())
        with np.load(output/'replay.npz',allow_pickle=False) as replay:
            lengths = np.diff(replay['episode_offsets'])
            keep = [i for i,(ep,step) in enumerate(wrapper.probe_tags) if step < lengths[ep]]
            tags = np.asarray([wrapper.probe_tags[i] for i in keep],dtype=np.int64)
            indices = replay['episode_offsets'][tags[:,0]] + tags[:,1]
            assert np.array_equal(indices,replay['probe_action_indices'])
            raw_probes = np.asarray(wrapper.raw_probes,dtype=np.uint8)[keep]
            raw_ep0 = np.asarray(wrapper.raw_episode0,dtype=np.uint8)
            transform = lambda rgb:np.clip(np.rint(rgb.astype(np.float32)*args.brightness),0,255).astype(np.uint8)
            assert np.array_equal(transform(raw_probes),replay['image_probes'])
            assert np.array_equal(transform(raw_ep0),replay['episode_0_images'])
        np.savez_compressed(output/'input_transform.npz',raw_probe_images=raw_probes,
                            raw_episode_0_images=raw_ep0,probe_tags=tags,probe_action_indices=indices)
        summary.update(brightness_factor=args.brightness,
                       input_transform_sha256=common.sha256(output/'input_transform.npz'),
                       brightness_evaluator_sha256=common.sha256(Path(__file__)),
                       scope='Fixed-model policy-image brightness diagnosis; no training or physical light change.',
                       image_storage='Transformed policy images in replay; raw matching ep0/probes in input_transform.npz.',
                       brightness_transform='float32 multiply, rint nearest even, clip0..255, uint8')
        summary['video']['content'] = 'Policy-observed transformed RGB, fixed episode0.'
        summary.update(initial_state_suite=args.suite,
                       official_nominal_initialization=args.suite == 'standard',
                       additional_training_source_sha256=extra_sources,
                       correction_evaluator_sha256=common.sha256(Path(__file__)),
                       collector_initialization_source_sha256=common.sha256(HERE / 'collect_corrections.py'))
        if args.suite == 'shifted':
            summary.update(initializations=initializations,
                           initial_state_rule='Official task2 reset, then true cube XY uniform +/-3cm,+/-6cm; cube velocity zero; robot and goal retained.',
                           scope='Custom initial-position stress suite; report separately from official nominal task2 results.')
        common.write_json(summary_path, summary)
    except Exception as error:
        common.write_json(output / 'error.json', {'status': 'failed', 'type': type(error).__name__,
                                                 'message': str(error)})
        raise
    finally:
        common.make_env = original_factory


if __name__ == '__main__':
    main()
