"""Reuse the frozen CPU evaluator with explicit standard/shifted initial states."""
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
    args = parser.parse_args()
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

    original_factory = common.make_env
    initializations = []

    class ShiftedReset(gym.Wrapper):
        def reset(self, *, seed=None, options=None):
            image, _, record = reset_episode(self.env, seed, shifted=True)
            initializations.append({'env_seed': seed, **record})
            return image, self.env.unwrapped.get_reset_info()

    if args.suite == 'shifted':
        common.make_env = lambda: ShiftedReset(original_factory())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        frozen_evaluator.evaluate(args, output)
        summary_path = output / 'summary.json'
        summary = json.loads(summary_path.read_text())
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
