#!/usr/bin/env python3
"""Run fixed exported policies in MuJoCo; small demonstrations are not score estimates.

Visual inputs are RGB + robot proprioception; true cube state is excluded.
Brightness changes policy pixels only. Shifted resets are a custom stress suite.
"""
import argparse
import csv
import json
from pathlib import Path
from portable_models import ROOT, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='visual_seed613_control')
    parser.add_argument('--output', type=Path, required=True, help='New output directory, never overwritten.')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--suite', choices=('standard', 'shifted'), default='standard')
    parser.add_argument('--brightness', type=float, choices=(1.0, 0.7, 0.55), default=1.0)
    parser.add_argument('--env-seed-start', type=int, default=150000)
    parser.add_argument('--action-seed-start', type=int, default=151000)
    args = parser.parse_args()
    if (args.episodes < 1 or min(args.env_seed_start, args.action_seed_start) < 0
            or max(args.env_seed_start, args.action_seed_start) + args.episodes > 2**32):
        parser.error('Require positive episodes and uint32 seeds.')
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import ogbench
    from visual.common import make_env, state_observation
    from visual.collect_corrections import reset_episode
    model, payload, reload_check = load_model(args.model, args.device)
    visual = payload['kind'] == 'visual_student'
    if not visual and args.brightness != 1.0:
        parser.error('Brightness applies to visual inputs only; state models receive no images.')
    args.output.mkdir(parents=True, exist_ok=False)
    env = make_env() if visual else ogbench.make_env_and_datasets('cube-single-play-singletask-v0', env_only=True)
    rows, writer = [], None
    try:
        if env.unwrapped._reward_task_id != 2 or env.spec.max_episode_steps != 200:
            raise ValueError('Expected OGBench task2, horizon200.')
        for episode in range(args.episodes):
            es, az = args.env_seed_start + episode, args.action_seed_start + episode
            if visual or args.suite == 'shifted':
                observation, state, initial = reset_episode(env, es, args.suite == 'shifted')
            else:
                observation, _ = env.reset(seed=es)
                state, initial = np.asarray(observation, dtype=np.float32), {}
            generator = torch.Generator(device='cpu').manual_seed(az)
            length, total_return, done = 0, 0.0, False
            if episode == 0:
                writer = imageio.get_writer(args.output / 'episode_000.mp4', fps=20, macro_block_size=1)
            while not done:
                noise = torch.randn((1, 5), generator=generator).to(args.device)
                if visual:
                    image = np.clip(np.rint(observation.astype(np.float32) * args.brightness), 0, 255).astype(np.uint8)
                    if writer is not None:
                        writer.append_data(image)
                    chw = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0).to(args.device)
                    proprio = torch.from_numpy(state[:19].copy()).unsqueeze(0).to(args.device)
                    with torch.inference_mode():
                        action = model(chw, proprio, noise)[0].cpu().numpy()
                else:
                    if writer is not None:
                        writer.append_data(np.asarray(env.render()))
                    with torch.inference_mode():
                        action = model(torch.from_numpy(state.copy()).unsqueeze(0).to(args.device), noise).clamp(-1, 1)[0].cpu().numpy()
                observation, reward, terminated, truncated, info = env.step(action)
                state = state_observation(env) if visual else np.asarray(observation, dtype=np.float32)
                length += 1
                total_return += float(reward)
                done = bool(terminated or truncated)
                if length >= 200 and not done:
                    raise ValueError('Expected termination or truncation by step200.')
            if writer is not None:
                final = (np.clip(np.rint(observation.astype(np.float32) * args.brightness), 0, 255).astype(np.uint8)
                         if visual else np.asarray(env.render()))
                writer.append_data(final)
                writer.close()
                writer = None
            row = {'episode': episode, 'env_seed': es, 'action_seed': az, 'success': int(info['success']),
                   'return': total_return, 'length': length, 'terminated': bool(terminated),
                   'truncated': bool(truncated), 'initialization': initial}
            rows.append(row)
            print(json.dumps(row), flush=True)
        with (args.output / 'episodes.csv').open('x', newline='') as stream:
            fields = [key for key in rows[0] if key != 'initialization']
            csv_writer = csv.DictWriter(stream, fieldnames=fields)
            csv_writer.writeheader()
            csv_writer.writerows({key: row[key] for key in fields} for row in rows)
        summary = {'scope': 'Independent small demo; not a new formal performance estimate.',
                   'model': args.model, 'reload_check': reload_check, 'episodes': args.episodes,
                   'successes': sum(row['success'] for row in rows), 'suite': args.suite,
                   'brightness': args.brightness, 'brightness_changes_physics': False,
                   'environment_steps': sum(row['length'] for row in rows), 'training_updates': 0,
                   'dataset_loaded': False, 'teacher_loaded': False, 'language_input': False,
                   'device': args.device, 'success_definition': 'Official task2 cube-goal distance <=4cm; no added release/hold condition.',
                   'video_selection': 'Episode0 regardless of outcome; visual video shows policy-observed pixels.',
                   'rows': rows}
        (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    finally:
        if writer is not None:
            writer.close()
        env.close()


if __name__ == '__main__':
    main()
