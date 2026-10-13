"""Recompute published statistics from public CSVs; does not run any model or simulator."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--output', type=Path)
args = parser.parse_args()
ROOT = args.root.resolve()


def read(rel):
    return json.loads((ROOT / rel).read_text())


def rows(rel):
    with (ROOT / rel).open(newline='') as stream:
        return list(csv.DictReader(stream))


def equal(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z*z/count
    center = (p + z*z/(2*count)) / denominator
    radius = z*math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
    return [max(0, center-radius), min(1, center+radius)]


def check_group(csv_rows, summary, n, interval):
    assert len(csv_rows) == summary['episodes'] == n
    assert [int(r['episode']) for r in csv_rows] == list(range(n))
    outcome = np.array([int(r['success']) for r in csv_rows])
    assert np.isin(outcome, [0, 1]).all()
    assert all(int(r['ever_success']) == int(r['success']) for r in csv_rows)
    assert int(outcome.sum()) == summary['successes']
    equal(outcome.mean(), summary['success_rate'])
    equal(wilson(int(outcome.sum()), n), summary[interval])
    if 'mean_return' in summary:
        equal(np.mean([float(r['return']) for r in csv_rows]), summary['mean_return'])
        equal(np.mean([int(r['length']) for r in csv_rows]), summary['mean_length'])
    return outcome


state = read('evidence/state/results.json')
state_means = {arm: [] for arm in ('raw', 'average5')}
for seed, checkpoints in state['formal_evaluations'].items():
    sums = {arm: [] for arm in state_means}
    for step, group in checkpoints.items():
        outcomes = {}
        for arm in sums:
            s = group[arm]
            outcomes[arm] = check_group(rows(s['public_episode_csv']), s, 100,
                                        'success_rate_wilson_95_percent_ci')
            sums[arm].append(float(outcomes[arm].mean()))
        diff = outcomes['average5'] - outcomes['raw']
        paired = group['paired']
        equal(diff.mean(), paired['success_rate_improvement'])
        rescued = int(((outcomes['raw'] == 0) & (outcomes['average5'] == 1)).sum())
        lost = int(((outcomes['raw'] == 1) & (outcomes['average5'] == 0)).sum())
        assert rescued == paired['raw_failure_candidate_success']
        assert lost == paired['raw_success_candidate_failure']
        idx = np.random.RandomState(paired['bootstrap_seed']).randint(
            0, 100, size=(paired['bootstrap_resamples'], 100))
        equal(np.quantile(diff[idx].mean(1), [.025, .975]),
              paired['paired_case_bootstrap_95_percent_ci'])
    for arm, rates in sums.items():
        mean = float(np.mean(rates))
        equal(mean, state['per_seed_last_three_mean'][seed][arm])
        state_means[arm].append(mean)
for arm, values in state_means.items():
    equal(np.mean(values), state['across_three_seeds'][arm]['mean'])
    equal(np.std(values, ddof=1), state['across_three_seeds'][arm]['sample_standard_deviation_ddof1'])

temporal = read('evidence/visual_temporal/results.json')
paired_rows = rows('evidence/visual_temporal/paired_cases.csv')
idx = np.random.default_rng(20261019).integers(0, 100, size=(10000, 100))
for suite in ('standard', 'shifted'):
    cases = [r for r in paired_rows if r['suite'] == suite]
    assert len(cases) == 100
    differences = []
    for seed in (613, 614, 615):
        outcomes = {}
        for arm in ('student_prefix', 'student_full'):
            name = f'seed{seed}_{arm}_{suite}'
            csv_rows = rows(f'evidence/visual_temporal/evaluations/{name}.csv')
            outcomes[arm] = check_group(csv_rows, temporal['groups'][name], 100, 'wilson95')
            assert all(r['env_seed'] == c['env_seed'] and r['action_seed'] == c['action_seed']
                       and int(r['success']) == int(c[f'seed{seed}_{arm}'])
                       for r, c in zip(csv_rows, cases))
        diff = outcomes['student_full'] - outcomes['student_prefix']
        differences.append(diff)
        item = temporal['paired'][suite]['per_seed'][str(seed)]
        equal(diff.mean()*100, item['difference_pp'])
        equal(np.quantile(diff[idx].mean(1)*100, [.025, .975]), item['bootstrap95_pp'])
    case_mean = np.stack(differences).mean(0)
    sampled = case_mean[idx].mean(1)*100
    item = temporal['paired'][suite]['fixed_three_seed_case_cluster_mean']
    equal(case_mean.mean()*100, item['difference_pp'])
    equal(np.quantile(sampled, [.025, .975]), item['bootstrap95_pp'])
    equal(np.quantile(sampled, [.0125, .9875]), item['bootstrap97_5_pp'])

brightness = read('evidence/brightness/results.json')
paired_rows = rows('evidence/brightness/paired_cases.csv')
streams = np.random.SeedSequence(20261021).spawn(2)
suite_samples, suite_means = {}, {}
for suite, stream in zip(('standard', 'shifted'), streams):
    cases = [r for r in paired_rows if r['suite'] == suite]
    assert len(cases) == 50
    idx = np.random.default_rng(stream).integers(0, 50, size=(10000, 50))
    suite_samples[suite], suite_means[suite] = {}, {}
    for condition in ('clean', 'dim07', 'dim055'):
        differences = []
        for seed in (613, 614, 615):
            outcomes = {}
            for arm in ('control', 'augmentation'):
                name = f'seed{seed}_{arm}_{condition}_{suite}'
                csv_rows = rows(f'evidence/brightness/evaluations/{name}.csv')
                outcomes[arm] = check_group(csv_rows, brightness['groups'][name], 50, 'wilson95')
                assert all(r['env_seed'] == c['env_seed'] and r['action_seed'] == c['action_seed']
                           and int(r['success']) == int(c[f'seed{seed}_{arm}_{condition}'])
                           for r, c in zip(csv_rows, cases))
            diff = outcomes['augmentation'] - outcomes['control']
            differences.append(diff)
            item = brightness['paired'][suite][condition]['per_seed'][str(seed)]
            equal(diff.mean()*100, item['difference_pp'])
            equal(np.quantile(diff[idx].mean(1)*100, [.025, .975]), item['paired95_pp'])
        case_mean = np.stack(differences).mean(0)
        sampled = case_mean[idx].mean(1)*100
        suite_samples[suite][condition] = sampled
        suite_means[suite][condition] = case_mean.mean()*100
        item = brightness['paired'][suite][condition]
        equal(case_mean.mean()*100, item['fixed_three_seed_mean_difference_pp'])
        equal(np.quantile(sampled, [.025, .975]), item['paired_case_cluster95_pp'])
for condition in ('clean', 'dim07', 'dim055'):
    item = brightness['macro'][condition]
    equal((suite_means['standard'][condition]+suite_means['shifted'][condition])/2, item['difference_pp'])
    sampled = (suite_samples['standard'][condition]+suite_samples['shifted'][condition])/2
    equal(np.quantile(sampled, [.025, .975]), item['paired_stratified_case_cluster95_pp'])
primary, clean = brightness['macro']['dim07'], brightness['macro']['clean']
assert brightness['decision']['primary_benefit_gate'] == (
    primary['difference_pp'] >= 10 and primary['paired_stratified_case_cluster95_pp'][0] > 0)
assert brightness['decision']['clean_cost_gate'] == (
    clean['difference_pp'] >= -5 and clean['paired_stratified_case_cluster95_pp'][0] >= -10)

benchmark = read('evidence/benchmark/results.json')
for item in benchmark['runs'].values():
    summary = item['summary']
    assert summary['updates_completed'] == 1000 and summary['warm_updates'] == 999
    equal(1/summary['warm_update_seconds_mean'], summary['warm_updates_per_second'])

plot_data = read('assets/plot_data.json')
for seed, arms in plot_data['state'].items():
    for arm, counts in arms.items():
        assert counts == [state['formal_evaluations'][seed][str(step)][arm]['successes']
                          for step in (800000, 900000, 1000000)]
for key, summary in [('temporal', temporal), ('brightness', brightness)]:
    for name, item in plot_data[key]['groups'].items():
        for field in ('successes', 'episodes', 'success_rate'):
            equal(item[field], summary['groups'][name][field])
    assert plot_data[key]['paired'] == summary['paired']
for mode, item in plot_data['benchmark'].items():
    equal(item['updates_per_second'], benchmark['runs'][mode]['summary']['warm_updates_per_second'])

provenance = read('evidence/provenance_manifest.json')
for item in provenance['records']:
    path = ROOT / item['derived_path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['derived_sha256'], str(path)
report = {'status': 'passed', 'CSV_groups': 66, 'episodes': 4800,
          'state_groups': 18, 'temporal_groups': 12, 'brightness_groups': 36,
          'paired_bootstrap_intervals_recomputed': True,
          'derived_file_sha256_checked': len(provenance['records']),
          'training_updates': 0, 'environment_steps': 0,
          'scope': 'Public episode-table statistics and derived file identities only; does not repeat historical model, noise, pixel, optimizer, or physical-success trajectory audits.'}
if args.output:
    args.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
