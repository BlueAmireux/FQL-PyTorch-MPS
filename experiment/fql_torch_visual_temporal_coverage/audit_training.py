"""CPU checkpoint/label validation for full-window and two fixed references."""
from pathlib import Path
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
freeze = read(OUT / 'source_freeze.json')
assert all(sha256(Path(p)) == h for p, h in protocol['frozen_existing_inputs'].items())
assert all(sha256(Path(p)) == h for p, h in freeze['inputs_sha256'].items())
prior = Path(protocol['data']['previous_stage'])
teacher, identity = load_teacher('cpu')
_, parent = load_student_checkpoint(Path(protocol['parent']['run']), 20000, 'cpu')
with np.load(prior / 'val/data.npz', allow_pickle=False) as data:
    prefix_keys = set(zip(data['pair_ids'][data['behavior_id'] == 0].tolist(),
                          data['step_index'][data['behavior_id'] == 0].tolist()))
with np.load(prior / 'raw_collection/val_student/data.npz', allow_pickle=False) as data:
    heldout_raw = {key: data[key] for key in ('images', 'states', 'proprio', 'episode_ids', 'step_index')}
prefix_mask = np.array([(int(ep), int(t)) in prefix_keys for ep, t in
                        zip(heldout_raw['episode_ids'], heldout_raw['step_index'])])
assert prefix_mask.sum() == 197 and (~prefix_mask).sum() == 267
runs = {'student_full': OUT / 'student_full', 'student_prefix': prior / 'student_only',
        'teacher_prefix': prior / 'teacher_only'}
records, segments, payloads = {}, {}, {}

def metrics(model, data, seed):
    rgb, states, proprio = (torch.from_numpy(data[key].copy()) for key in ('images', 'states', 'proprio'))
    noise = torch.randn((len(rgb), 5), generator=torch.Generator().manual_seed(seed))
    action_errors, xyz_errors = [], []
    with torch.inference_mode():
        for start in range(0, len(rgb), 256):
            stop = min(start + 256, len(rgb))
            visual = model.encoder(rgb[start:stop].float() / 255)
            actual = model.actor(torch.cat((proprio[start:stop], visual), -1), noise[start:stop])
            target = teacher(states[start:stop], noise[start:stop])
            assert all(torch.isfinite(t).all() for t in (actual, target, visual))
            action_errors.append((actual.clamp(-1, 1) - target.clamp(-1, 1)).square())
            xyz_errors.append((visual[:, :3] - states[start:stop, 19:22]).square())
    action, xyz = torch.cat(action_errors), torch.cat(xyz_errors)
    masks = {'all': np.ones(len(rgb), dtype=bool)}
    if seed == 1717:
        masks.update(prefix=prefix_mask, excluded_late=~prefix_mask)
    return {name: {'samples': int(mask.sum()), 'clipped_mse': action[mask].mean().item(),
                   'xyz_euclidean_rmse_cm': (xyz[mask].sum(1).mean().sqrt() * 10).item()}
            for name, mask in masks.items()}

for arm, run in runs.items():
    model, payload = load_student_checkpoint(run, 40000, 'cpu')
    config, segment = read(run / 'config.json'), read(run / 'segment_20000_40000.json')
    assert payload['config'] == config and config['parent_checkpoint_sha256'] == protocol['parent']['sha256']
    assert payload['teacher']['sha256'] == identity['sha256'] and segment['teacher_parameters_unchanged']
    assert segment['start_step'] == 20000 and segment['additional_updates'] == 20000
    assert config['batch_size'] == 256 and config['learning_rate'] == 1e-4 and config['xyz_loss_weight'] == 1
    for name, digest in config['additional_source_sha256'].items():
        assert sha256(ROOT / '复现/fql/pytorch/visual' / name) == digest
    assert payload['optimizer']['param_groups'] == parent['optimizer']['param_groups']
    assert all(torch.isfinite(t).all() for t in payload['model'].values())
    for state in payload['optimizer']['state'].values():
        assert all(torch.isfinite(t).all() for t in state.values() if isinstance(t, torch.Tensor))
        assert int(state['step'].item()) == 40000
    assert segment['first_update']['all_gradients_finite'] and segment['first_update']['teacher_gradients_absent']
    assert segment['first_update']['encoder_max_abs_parameter_change'] > 0
    category, behavior = ('teacher', 1) if arm == 'teacher_prefix' else ('student', 0)
    assert config['selected_new_pool'] == {'category': category, 'behavior_id': behavior, 'unique_samples': 740, 'episodes': 20}
    _, early = load_student_checkpoint(run, 21000, 'cpu')
    assert early['config'] == config
    if arm == 'student_full':
        assert config['matching']['student_window'] == 'whole_raw_trajectory_uniform_subsample'
    validation = {}
    for name, path, seed in [('old', ROOT / 'experiment/fql_torch_visual_feasibility/paired_data_v2/val.npz', 614),
                             ('new', prior / 'val/data.npz', 1617)]:
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in ('images', 'states', 'proprio')}
        actual = metrics(model, arrays, seed)['all']
        expected = segment['final_validation'][name]['normal']
        assert abs(actual['clipped_mse'] - expected['clipped_mse']) < 1e-5
        assert abs(actual['xyz_euclidean_rmse_cm'] - expected['xyz_euclidean_rmse_cm']) < 1e-3
        validation[name] = actual
    records[arm] = {'checkpoint_sha256': sha256(run / 'checkpoint_40000.pt'),
                    'cpu_reload_error': payload['reload_max_abs_error'], 'cpu_early_reload_error': early['reload_max_abs_error'],
                    'optimizer_steps40000_and_parameters_finite': True,
                    'teacher_unchanged': True, 'CPU_heldout_recomputation': validation,
                    'raw_heldout_student_same_noise1717': metrics(model, heldout_raw, 1717),
                    'first1000': segment['first_1000'], 'training_seconds': segment['training_seconds'],
                    'memory': segment['memory']}
    segments[arm], payloads[arm] = segment, payload
reference = payloads['student_prefix']
for arm, payload in payloads.items():
    assert segments[arm]['initial_validation'] == segments['student_prefix']['initial_validation']
    assert all(torch.equal(payload['rng'][key], reference['rng'][key]) for key in ('cpu', 'mps'))
    assert payload['sampler'] == reference['sampler']
report = {'status': 'passed', 'arms': records, 'same_parent_Adam_and_initial_validation': True,
          'same_final_CPU_MPS_rng': True, 'same_old_and_new_sampler_end_states': True,
          'raw_heldout_diagnostic_noise_seed': 1717, 'new_supervised_updates': 20000,
          'new_FQL_RL_updates': 0, 'audit_training_updates': 0, 'audit_environment_steps': 0,
          'audit_source_sha256': sha256(Path(__file__))}
with (OUT / 'training_acceptance.json').open('x') as stream:
    json.dump(report, stream, indent=2); stream.write('\n')
print(json.dumps({'status': 'passed', 'arms': list(records),
                  'reload_errors': {name: value['cpu_reload_error'] for name, value in records.items()}}))
