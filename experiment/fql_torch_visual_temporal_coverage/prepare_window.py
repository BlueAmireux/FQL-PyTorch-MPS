"""Preserve per-case frame counts; change only the student sampling window."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

OUT = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
protocol = read(OUT / 'protocol.json')
prior = Path(protocol['data']['previous_stage'])
assert all(sha(Path(p)) == h for p, h in protocol['frozen_existing_inputs'].items())
old_manifest = read(prior / 'train/manifest.json')
raw_manifest = read(prior / 'raw_collection/train_student/manifest.json')
assert sha(prior / 'train/data.npz') == old_manifest['data_sha256']
assert sha(prior / 'raw_collection/train_student/data.npz') == raw_manifest['data_sha256']
with np.load(prior / 'train/data.npz', allow_pickle=False) as archive:
    previous = {key: archive[key] for key in archive.files}
with np.load(prior / 'raw_collection/train_student/data.npz', allow_pickle=False) as archive:
    raw = {key: archive[key] for key in archive.files}
with (prior / 'train/pairs.csv').open(newline='') as stream:
    pairs = list(csv.DictReader(stream))
assert len(pairs) == 20
chosen, rows = [], []
for pair, record in enumerate(pairs):
    available = np.flatnonzero(raw['episode_ids'] == pair)
    prefix = np.flatnonzero((previous['behavior_id'] == 0) & (previous['pair_ids'] == pair))
    count = int(record['frames_per_behavior'])
    assert len(prefix) == count and 1 <= count <= len(available)
    relative = np.rint(np.linspace(0, len(available) - 1, count)).astype(np.int64)
    assert len(np.unique(relative)) == count
    selected = available[relative]
    assert selected[0] == available[0] and selected[-1] == available[-1]
    assert np.array_equal(raw['images'][selected[0]], previous['images'][prefix[0]])
    assert np.array_equal(raw['states'][selected[0]], previous['states'][prefix[0]])
    assert int(raw['env_seeds'][pair]) == int(record['env_seed'])
    late = raw['step_index'][selected] >= int(record['common_length'])
    overlap = np.isin(raw['step_index'][selected], previous['step_index'][prefix])
    chosen.extend(selected.tolist())
    rows.append({'pair': pair, 'env_seed': int(record['env_seed']), 'action_seed': int(record['action_seed']),
                 'shifted': bool(pair % 2), 'raw_student_frames': len(available), 'frames_kept': count,
                 'prefix_end_step_exclusive': int(record['common_length']),
                 'full_last_step': int(raw['step_index'][selected[-1]]),
                 'late_frames_included': int(late.sum()), 'prefix_frame_overlap': int(overlap.sum())})
chosen = np.asarray(chosen, dtype=np.int32)
teacher_ids = np.flatnonzero(previous['behavior_id'] == 1).astype(np.int32)
assert len(chosen) == len(teacher_ids) == 740 and len(np.unique(chosen)) == 740
keys = ('images', 'states', 'proprio', 'episode_ids', 'behavior_id', 'shifted', 'step_index', 'noises', 'executed_actions', 'teacher_raw_actions')
merged = {key: np.concatenate((raw[key][chosen], previous[key][teacher_ids])) for key in keys}
merged['pair_ids'] = np.concatenate((raw['episode_ids'][chosen], previous['pair_ids'][teacher_ids]))
for key in ('env_seeds', 'action_seeds', 'terminal_states'):
    merged[key] = previous[key].copy()
for key in (*keys, 'pair_ids'):
    assert np.array_equal(merged[key][740:], previous[key][teacher_ids])
assert np.array_equal(merged['proprio'], merged['states'][:, :19])
assert all(np.isfinite(merged[key]).all() for key in ('states', 'noises', 'executed_actions', 'teacher_raw_actions'))
assert np.array_equal(np.unique(merged['episode_ids']), np.arange(40))
assert np.array_equal(merged['pair_ids'][:740], merged['pair_ids'][740:])
target = OUT / 'train'
target.mkdir(exist_ok=False)
with (target / 'data.npz').open('xb') as stream:
    np.savez_compressed(stream, **merged)
with (target / 'selection_indices.npz').open('xb') as stream:
    np.savez_compressed(stream, student_raw_indices=chosen, teacher_matched_indices=teacher_ids)
with (target / 'pairs.csv').open('x', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
sources = dict(old_manifest['source_sha256']); sources[str(Path(__file__))] = sha(Path(__file__))
manifest = {'status': 'completed', 'split': 'train', 'purpose': 'Equal-case equal-count whole-student-window intervention',
            'paired_cases': 20, 'completed_episodes': 40, 'samples': 1480, 'samples_per_behavior': 740,
            'teacher': old_manifest['teacher'], 'student': old_manifest['student'], 'source_sha256': sources,
            'data_sha256': sha(target / 'data.npz'), 'selection_indices_sha256': sha(target / 'selection_indices.npz'),
            'pairs_csv_sha256': sha(target / 'pairs.csv'), 'student_replay_window': 'whole_raw_trajectory_uniform_subsample',
            'teacher_pool_exact_copy': True, 'selection_by_success': False,
            'matching_rule': protocol['data']['student_pool'], 'limitations': protocol['limitations'],
            'source_previous_train_manifest_sha256': sha(prior / 'train/manifest.json'),
            'source_raw_student_manifest_sha256': sha(prior / 'raw_collection/train_student/manifest.json')}
(target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
report = {'status': 'passed', 'same20_cases_and740_frames': True, 'same_per_case_counts_and_order': True,
          'teacher_subset_all_fields_exact_copy': True, 'student_all_fields_exact_raw_subset': True,
          'initial_rgb_and_state_unchanged': True, 'selection_by_success': False,
          'prefix_frames_retained': sum(r['prefix_frame_overlap'] for r in rows),
          'previously_excluded_late_frames_included': sum(r['late_frames_included'] for r in rows),
          'new_collection_episodes': 0, 'new_environment_steps': 0,
          'data_sha256': manifest['data_sha256'], 'manifest_sha256': sha(target / 'manifest.json'),
          'processing_source_sha256': sha(Path(__file__))}
assert report['prefix_frames_retained'] + report['previously_excluded_late_frames_included'] == 740
assert all(sha(Path(p)) == h for p, h in protocol['frozen_existing_inputs'].items())
with (OUT / 'data_acceptance.json').open('x') as stream:
    json.dump(report, stream, indent=2); stream.write('\n')
print(json.dumps(report), flush=True)
