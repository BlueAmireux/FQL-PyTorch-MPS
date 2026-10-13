"""Match whole initial cases and common pre-action time indices; audit raw labels."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / '复现/fql/pytorch'))
from visual.common import load_student_checkpoint, load_teacher, sha256

torch.set_num_threads(4)
read = lambda p: json.loads(p.read_text())
protocol = read(OUT / 'protocol.json')
assert all(sha256(Path(p)) == h for p, h in protocol['frozen_existing_inputs'].items())
student, payload = load_student_checkpoint(Path(protocol['parent']['run']), 20000, 'cpu')
teacher, identity = load_teacher('cpu')
assert identity['sha256'] == payload['teacher']['sha256']
report = {'status': 'passed', 'split_unit': 'whole paired initial cases', 'splits': {},
          'new_environment_steps': 0, 'new_training_updates': 0}

for split in ('train', 'val'):
    case = protocol['collection']['cases'][split]
    n = case['pairs']
    raw, manifests, rows, selected = {}, {}, {}, {}
    pair_rows = []
    errors = {}
    for behavior, name in enumerate(('student', 'teacher')):
        source = OUT / 'raw_collection' / f'{split}_{name}'
        manifest = read(source / 'manifest.json')
        assert manifest['status'] == 'completed' and manifest['split'] == split
        assert manifest['completed_episodes'] == n and manifest['sample_stride'] == 2
        assert manifest['teacher']['sha256'] == identity['sha256']
        assert manifest['student']['checkpoint_sha256'] == protocol['parent']['sha256']
        assert sha256(source / 'data.npz') == manifest['data_sha256']
        assert sha256(source / 'episodes.csv') == manifest['episodes_csv_sha256']
        assert all(sha256(Path(p)) == h for p, h in manifest['source_sha256'].items())
        with np.load(source / 'data.npz', allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        with (source / 'episodes.csv').open(newline='') as stream:
            group_rows = list(csv.DictReader(stream))
        assert len(group_rows) == n
        assert np.array_equal(data['env_seeds'], np.arange(case['env_seed_start'], case['env_seed_start'] + n))
        assert np.array_equal(data['action_seeds'], np.arange(case['action_seed_start'], case['action_seed_start'] + n))
        assert np.array_equal(data['proprio'], data['states'][:, :19])
        assert np.array_equal(np.unique(data['episode_ids']), np.arange(n))
        assert np.all(data['behavior_id'] == behavior)
        assert all(np.isfinite(data[key]).all() for key in ('states', 'proprio', 'noises', 'executed_actions', 'teacher_raw_actions', 'terminal_states'))
        label_error = action_error = 0.
        with torch.inference_mode():
            for start in range(0, len(data['images']), 256):
                states = torch.from_numpy(data['states'][start:start + 256].copy())
                noise = torch.from_numpy(data['noises'][start:start + 256].copy())
                target = teacher(states, noise)
                action = target.clamp(-1, 1) if behavior else student(
                    torch.from_numpy(data['images'][start:start + 256].copy()), states[:, :19], noise)
                assert torch.isfinite(target).all() and torch.isfinite(action).all()
                label_error = max(label_error, float(np.abs(target.numpy() - data['teacher_raw_actions'][start:start + 256]).max()))
                action_error = max(action_error, float(np.abs(action.numpy() - data['executed_actions'][start:start + 256]).max()))
        assert max(label_error, action_error) <= 1e-4
        for ep, row in enumerate(group_rows):
            ids = np.flatnonzero(data['episode_ids'] == ep)
            length = int(row['length'])
            assert np.array_equal(data['step_index'][ids], np.arange(0, length, 2))
            generator = torch.Generator().manual_seed(case['action_seed_start'] + ep)
            noise = torch.cat([torch.randn((1, 5), generator=generator) for _ in range(length)]).numpy()
            assert np.array_equal(data['noises'][ids], noise[::2])
            assert np.all(data['shifted'][ids] == bool(ep % 2))
            assert row['robot_goal_ctrl_unchanged'] == 'True' and float(row['initial_contact']) == 0
            assert float(row['initial_goal_distance_m']) > .06
            shift = np.random.default_rng(case['env_seed_start'] + ep + 1000000).uniform([-.03, -.06], [.03, .06]) if ep % 2 else np.zeros(2)
            assert np.array_equal(shift, [float(row['dx']), float(row['dy'])])
            terminal_distance = np.linalg.norm(data['terminal_states'][ep, 19:22] / 10 + [.425, 0., 0.] - np.array([.5, 0., .02]))
            success = int(terminal_distance <= .04)
            assert success == int(row['success']) == int(row['ever_success'])
            assert float(row['return']) == -length + success
            assert (row['terminated'] == 'True') == bool(success)
            assert (row['truncated'] == 'True') or success
            assert length <= 200 and int(row['samples']) == len(ids)
        raw[name], manifests[name], rows[name], selected[name] = data, manifest, group_rows, []
        errors[name] = {'teacher_raw_label_max_abs_error': label_error, 'executed_action_max_abs_error': action_error,
                        'raw_samples': len(data['images']), 'successes': sum(int(r['success']) for r in group_rows)}
    for ep in range(n):
        ids = {name: np.flatnonzero(data['episode_ids'] == ep) for name, data in raw.items()}
        a, b = (raw[name] for name in ('student', 'teacher'))
        first_a, first_b = ids['student'][0], ids['teacher'][0]
        assert np.array_equal(a['states'][first_a], b['states'][first_b])
        assert np.array_equal(a['images'][first_a], b['images'][first_b])
        lengths = {name: int(rows[name][ep]['length']) for name in raw}
        common_length = min(lengths.values())
        common_times = np.arange(0, common_length, 2)
        kept = {name: value[raw[name]['step_index'][value] < common_length] for name, value in ids.items()}
        for name, value in kept.items():
            assert np.array_equal(raw[name]['step_index'][value], common_times)
            selected[name].extend(value.tolist())
        assert np.array_equal(a['noises'][kept['student']], b['noises'][kept['teacher']])
        pair_rows.append({'pair': ep, 'env_seed': case['env_seed_start'] + ep, 'action_seed': case['action_seed_start'] + ep,
                          'shifted': bool(ep % 2), 'student_length': lengths['student'], 'teacher_length': lengths['teacher'],
                          'common_length': common_length, 'frames_per_behavior': len(common_times),
                          'student_success': int(rows['student'][ep]['success']), 'teacher_success': int(rows['teacher'][ep]['success']),
                          'student_late_frames_excluded': len(ids['student']) - len(common_times),
                          'teacher_late_frames_excluded': len(ids['teacher']) - len(common_times)})
    keys = ('images', 'states', 'proprio', 'episode_ids', 'behavior_id', 'shifted', 'step_index', 'noises', 'executed_actions', 'teacher_raw_actions')
    merged = {key: np.concatenate([raw[name][key][selected[name]] for name in ('student', 'teacher')]) for key in keys}
    merged['pair_ids'] = merged['episode_ids'].copy()
    merged['episode_ids'][len(selected['student']):] += n
    for key in ('env_seeds', 'action_seeds', 'terminal_states'):
        merged[key] = np.concatenate([raw[name][key] for name in ('student', 'teacher')])
    assert len(selected['student']) == len(selected['teacher']) > 0
    target = OUT / split
    target.mkdir(exist_ok=False)
    with (target / 'data.npz').open('xb') as stream:
        np.savez_compressed(stream, **merged)
    with (target / 'pairs.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0]))
        writer.writeheader(); writer.writerows(pair_rows)
    sources = dict(manifests['student']['source_sha256'])
    assert sources == manifests['teacher']['source_sha256']
    sources[str(Path(__file__))] = sha256(Path(__file__))
    manifest = {'status': 'completed', 'split': split, 'purpose': 'Equal-size same-case same-time-prefix replay comparison',
                'paired_cases': n, 'completed_episodes': 2 * n, 'samples': len(merged['images']),
                'samples_per_behavior': len(selected['student']), 'sample_stride': 2,
                'teacher': identity, 'student': manifests['student']['student'], 'source_sha256': sources,
                'data_sha256': sha256(target / 'data.npz'), 'pairs_csv_sha256': sha256(target / 'pairs.csv'),
                'raw_collections': {name: {'path': str(OUT / 'raw_collection' / f'{split}_{name}'),
                                          'manifest_sha256': sha256(OUT / 'raw_collection' / f'{split}_{name}' / 'manifest.json'),
                                          'data_sha256': manifests[name]['data_sha256']} for name in raw},
                'matching_rule': protocol['collection']['matching'], 'selection_by_success': False,
                'terminal_states': 'Raw whole-rollout terminal states for audit only; may lie after the common training window.',
                'limitations': ['Common prefix length is outcome-dependent; late failed-student states are censored.']}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    report['splits'][split] = {'pairs': n, 'matched_frames_per_behavior': len(selected['student']),
                             'standard_frames_per_behavior': sum(r['frames_per_behavior'] for r in pair_rows if not r['shifted']),
                             'shifted_frames_per_behavior': sum(r['frames_per_behavior'] for r in pair_rows if r['shifted']),
                             'raw_audit': errors, 'initial_state_and_rgb_exact_match': True,
                             'matched_time_indices_and_noise_exact_match': True,
                             'late_frames_excluded': {name: len(raw[name]['images']) - len(selected[name]) for name in raw},
                             'manifest_sha256': sha256(target / 'manifest.json'), 'data_sha256': manifest['data_sha256']}
assert not set(np.unique(np.load(OUT / 'train/data.npz')['env_seeds'])) & set(np.unique(np.load(OUT / 'val/data.npz')['env_seeds']))
assert all(sha256(Path(p)) == h for p, h in protocol['frozen_existing_inputs'].items())
report.update(train_val_cases_disjoint=True, frozen_existing_inputs_unchanged=True, audit_source_sha256=sha256(Path(__file__)))
with (OUT / 'data_acceptance.json').open('x') as stream:
    json.dump(report, stream, indent=2); stream.write('\n')
print(json.dumps(report), flush=True)
