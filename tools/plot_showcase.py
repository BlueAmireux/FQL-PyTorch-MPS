"""Render release figures from existing audited results; no training or simulation."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
args = parser.parse_args()
ROOT = args.root.resolve()
OUT = ROOT / 'assets'
OUT.mkdir(parents=True, exist_ok=True)
BLUE, TEAL, ORANGE, GRAY = '#355C9A', '#087F8C', '#C77928', '#64748B'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'axes.labelcolor': '#243447',
                     'text.color': '#243447', 'savefig.facecolor': 'white',
                     'svg.fonttype': 'none'})
sources, figures = {}, []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, name, caption):
    fig.text(.05, .02, caption, fontsize=9, va='bottom', color=GRAY)
    for ext in ('png', 'svg'):
        path = OUT / f'{name}.{ext}'
        fig.savefig(path, dpi=200)
        figures.append({'path': path.relative_to(ROOT).as_posix(),
                        'sha256': digest(path), 'bytes': path.stat().st_size,
                        'caption': caption})
    plt.close(fig)


data = json.loads((OUT / 'plot_data.json').read_text())
state = {'formal_evaluations': {}, 'per_seed_last_three_mean': {}}
for seed, arms in data['state'].items():
    state['formal_evaluations'][seed] = {
        str(step): {arm: {'successes': values[i]} for arm, values in arms.items()}
        for i, step in enumerate((800000, 900000, 1000000))}
    state['per_seed_last_three_mean'][seed] = {arm: float(np.mean(values)) / 100 for arm, values in arms.items()}
temporal, brightness = data['temporal'], data['brightness']
benchmark = {'runs': {mode: {'summary': {'warm_updates_per_second': item['updates_per_second'],
              'memory_after': item['memory_after']}, 'gpu_driver_proxy': {'devices': {'proxy': item['whole_device_proxy']}}}
              for mode, item in data['benchmark'].items()}}

# The terminal metric and the reported last-three-checkpoint metric are distinct.
fig = plt.figure(figsize=(14.5, 9))
grid = fig.add_gridspec(2, 3, height_ratios=[1.1, 1], hspace=.55)
for seed in range(3):
    ax = fig.add_subplot(grid[0, seed])
    data['state'][str(seed)] = {}
    for arm, color, marker, label in [('raw', BLUE, 'o', 'Original actor'),
                                     ('average5', TEAL, 's', '5-snapshot parameter mean')]:
        counts = [state['formal_evaluations'][str(seed)][str(step)][arm]['successes']
                  for step in (800000, 900000, 1000000)]
        data['state'][str(seed)][arm] = counts
        ax.plot([800, 900, 1000], counts, marker=marker, lw=2.5, color=color, label=label)
        for x, y in zip([800, 900, 1000], counts):
            offset = 1.2 if arm == 'average5' else -2.4
            ax.text(x, y + offset, str(y), ha='center', color=color, fontsize=10)
    ax.set(title=f'Training seed {seed}', xlabel='Offline updates (thousands)',
           ylim=(80, 105), xticks=[800, 900, 1000], yticks=[80, 85, 90, 95, 100])
    ax.grid(axis='y', alpha=.2)
    if seed == 0:
        ax.set_ylabel('Successes / 100 episodes')
handles, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .925), ncol=2, frameon=False)
ax = fig.add_subplot(grid[1, :])
metrics = []
for name in ('Last-three-point mean', 'Final 1M checkpoint'):
    for arm in ('raw', 'average5'):
        vals = ([state['per_seed_last_three_mean'][str(s)][arm] * 100 for s in range(3)]
                if name.startswith('Last') else [data['state'][str(s)][arm][-1] for s in range(3)])
        metrics.append({'metric': name, 'arm': arm, 'per_seed_percent': vals,
                        'mean_percent': float(np.mean(vals))})
data['state_aggregates'] = metrics
positions = [0, .65, 2, 2.65]
for x, item in zip(positions, metrics):
    color = BLUE if item['arm'] == 'raw' else TEAL
    ax.bar(x, item['mean_percent'], width=.52, color=color, alpha=.85)
    ax.scatter(np.array([x-.09, x, x+.09]), item['per_seed_percent'], color='#172B4D', s=25, zorder=3)
    ax.text(x, item['mean_percent'] + 2, f"{item['mean_percent']:.2f}%", ha='center', weight='bold')
ax.set(ylim=(0, 108), ylabel='Success rate (%)', xticks=[.325, 2.325],
       xticklabels=['Mean over 800k / 900k / 1M, then three seeds',
                    '1M only, then three seeds'])
ax.grid(axis='y', alpha=.2)
ax.text(.02, .1, 'Dots: three training seeds (descriptive spread; no confidence interval)',
        transform=ax.transAxes, fontsize=10, color='white',
        bbox={'facecolor': '#243447', 'alpha': .8, 'edgecolor': 'none', 'pad': 4})
fig.suptitle('State FQL: preserve the terminal regression and report the deployment alternative',
             fontsize=17, x=.5, y=.98)
fig.subplots_adjust(top=.84, bottom=.16, left=.07, right=.97)
save(fig, 'state_results',
     'Each point uses 100 fixed evaluation episodes; the seed-0 fold was also used during development.\n'
     'Parameter averaging is inference-only and adds no RL updates. Three seeds do not reproduce the paper\'s eight-seed protocol.')

# A genuinely matched comparison: same frame count, starts, budget, and new cases.
fig, axes = plt.subplots(2, 2, figsize=(14, 9), gridspec_kw={'height_ratios': [1.1, 1]})
for col, suite in enumerate(('standard', 'shifted')):
    ax = axes[0, col]
    for arm, offset, color, label in [('student_prefix', -.18, BLUE, 'Common-prefix frames'),
                                      ('student_full', .18, TEAL, 'Full-trajectory frames')]:
        vals = [temporal['groups'][f'seed{s}_{arm}_{suite}']['successes'] for s in (613, 614, 615)]
        ax.bar(np.arange(3)+offset, vals, .34, color=color, label=label)
        for x, y in zip(np.arange(3)+offset, vals):
            ax.text(x, y+2, str(y), ha='center', color=color, weight='bold')
    ax.set(title='Standard starts' if suite == 'standard' else 'Initial XY displacement',
           ylim=(0, 108), xticks=np.arange(3), xticklabels=['Seed 613', 'Seed 614', 'Seed 615'])
    ax.grid(axis='y', alpha=.2)
    if col == 0:
        ax.set_ylabel('Successes / 100 episodes')
    delta_ax = axes[1, col]
    d = temporal['paired'][suite]
    for y, seed in enumerate((613, 614, 615)):
        item = d['per_seed'][str(seed)]
        val, ci = item['difference_pp'], item['bootstrap95_pp']
        delta_ax.errorbar(val, y, xerr=[[val-ci[0]], [ci[1]-val]], fmt='o',
                          color=GRAY, capsize=4, lw=1.7)
        delta_ax.text(val, y-.22, f'{val:+.0f} pp', ha='center', va='center', color=GRAY)
    m = d['fixed_three_seed_case_cluster_mean']
    val, ci = m['difference_pp'], m['bootstrap97_5_pp']
    delta_ax.errorbar(val, 3, xerr=[[val-ci[0]], [ci[1]-val]], fmt='D',
                      color=TEAL, capsize=5, lw=2.5, ms=8)
    delta_ax.text(val, 3-.22, f'{val:+.0f} pp', ha='center', va='center', color=TEAL, weight='bold')
    delta_ax.axvline(0, color=GRAY, ls='--', lw=1)
    delta_ax.set(yticks=np.arange(4), yticklabels=['613: 95% CI', '614: 95% CI', '615: 95% CI',
                                                  'Mean: 97.5% CI'],
                 xlabel='Full minus prefix success rate (percentage points)',
                 xlim=(-12, 38), ylim=(3.6, -.6))
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .92), ncol=2, frameon=False)
fig.suptitle('Visual temporal coverage: a controlled benefit with uneven seed effects', fontsize=17, y=.98)
fig.subplots_adjust(top=.83, bottom=.16, left=.14, right=.97, hspace=.45, wspace=.32)
save(fig, 'visual_temporal_coverage',
     'Same 740 student frames from 20 starts; same 40k endpoint. 1,200 episodes on 100 new cases per suite.\n'
     'Intervals resample shared case clusters, conditional on these three trained models; they are not training-seed population intervals.\n'
     'The full window also reduces early-frame density, so the causal mechanism is not isolated.')

# Robustness benefit must be shown beside the clean-performance cost.
fig = plt.figure(figsize=(14, 9))
grid = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=.5)
conditions = [('clean', 'Normal ×1.0'), ('dim07', 'Target dim ×0.7'), ('dim055', 'Outside range ×0.55')]
for col, suite in enumerate(('standard', 'shifted')):
    ax = fig.add_subplot(grid[0, col])
    for arm, offset, color, label in [('control', -.19, BLUE, 'Ordinary continuation'),
                                      ('augmentation', .19, ORANGE, 'Brightness augmentation')]:
        vals = [brightness['paired'][suite][c]['model_rate_mean_percent'][arm] for c, _ in conditions]
        ax.bar(np.arange(3)+offset, vals, .36, color=color, label=label, alpha=.86)
        for i, ((c, _), val) in enumerate(zip(conditions, vals)):
            seeds = [brightness['groups'][f'seed{s}_{arm}_{c}_{suite}']['success_rate']*100
                     for s in (613, 614, 615)]
            ax.scatter(np.array([i+offset-.07, i+offset, i+offset+.07]), seeds,
                       s=24, color='#172B4D', zorder=3)
            ax.text(i+offset, val+5, f'{val:.1f}', ha='center', color=color, weight='bold')
    ax.set(title='Standard starts' if suite == 'standard' else 'Initial XY displacement',
           ylim=(-3, 112), xticks=np.arange(3), xticklabels=[x[1] for x in conditions])
    ax.grid(axis='y', alpha=.2)
    if col == 0:
        ax.set_ylabel('Mean success rate (%)')
ax = fig.add_subplot(grid[1, :])
for i, (c, label) in enumerate(conditions):
    val = brightness['macro'][c]['difference_pp']
    ci = brightness['macro'][c]['paired_stratified_case_cluster95_pp']
    color = '#AD4853' if c == 'clean' else ORANGE
    ax.errorbar(val, i, xerr=[[val-ci[0]], [ci[1]-val]], fmt='o', ms=8, lw=2.5,
                capsize=5, color=color)
    ax.text(val, i-.24, f'{val:+.1f} pp  [{ci[0]:+.1f}, {ci[1]:+.1f}]',
            ha='center', va='center', color=color, weight='bold', fontsize=10)
ax.axvline(0, color=GRAY, ls='--', lw=1)
ax.set(yticks=range(3), yticklabels=[x[1] for x in conditions], ylim=(2.6, -.6), xlim=(-25, 87),
       xlabel='Augmentation minus control: equal-weight suite and model mean (percentage points)')
handles, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .93), ncol=2, frameon=False)
fig.suptitle('Brightness adaptation: dim-image gains and a failed clean-performance gate', fontsize=17, y=.98)
fig.subplots_adjust(top=.84, bottom=.17, left=.13, right=.97, wspace=.24)
save(fig, 'brightness_tradeoff',
     'Three seeds × two arms, each +20k supervised updates to 60k; 1,800 episodes on 100 shared physical initial cases.\n'
     'Top dots are per-seed rates; bottom whiskers are paired 95% case-cluster intervals conditional on the six fixed models.\n'
     'Synthetic RGB multiplication only. The clean-cost gate failed; the augmented models do not replace the normal models.')

# Separate process memory and shared-device driver snapshots; do not add them.
modes = ['jax_metal', 'torch_eager', 'torch_compiled_fused']
names = ['JAX / Metal', 'Torch / MPS\neager', 'Torch / MPS\ncompiled + fused']
for mode in modes:
    r = benchmark['runs'][mode]
    proxy = next(iter(r['gpu_driver_proxy']['devices'].values()))
    data['benchmark'][mode] = {'updates_per_second': r['summary']['warm_updates_per_second'],
                              'memory_after': r['summary']['memory_after'], 'whole_device_proxy': proxy}
fig, axes = plt.subplots(1, 3, figsize=(15, 6.8))
colors = [TEAL, GRAY, BLUE]
for i, mode in enumerate(modes):
    r = data['benchmark'][mode]
    ax = axes[0]
    v = r['updates_per_second']
    ax.bar(i, v, color=colors[i], width=.65)
    ax.text(i, v+1.5, f'{v:.2f}', ha='center', weight='bold')
    ax = axes[1]
    for off, key, alpha in [(-.18, 'rss_bytes', 1), (.18, 'peak_rss_bytes', .4)]:
        v = r['memory_after'][key]/2**20
        ax.bar(i+off, v, .34, color=colors[i], alpha=alpha,
               label=('End RSS' if key == 'rss_bytes' else 'Peak RSS') if i == 0 else None)
        ax.text(i+off, v+18, f'{v:.0f}', ha='center', fontsize=9)
    ax = axes[2]
    p = r['whole_device_proxy']; v = p['median_percent']
    ax.errorbar(i, v, yerr=[[v-p['p10_percent']], [p['p90_percent']-v]], fmt='o',
                color=colors[i], ms=10, capsize=6, lw=2)
    ax.text(i, v+5, f"{v:.0f}% (n={p['samples']})", ha='center', fontsize=10)
for ax in axes:
    ax.set(xticks=range(3), xticklabels=names, xlim=(-.6, 2.6))
    ax.grid(axis='y', alpha=.2)
axes[0].set(title='Synchronized warm throughput', ylabel='Updates / second', ylim=(0, 75))
axes[1].set(title='Process resident memory', ylabel='MiB', ylim=(0, 1100))
axes[1].legend(frameon=False, fontsize=10)
axes[2].set(title='Whole-device driver proxy', ylabel='Median and 10–90% range (%)', ylim=(0, 105))
fig.suptitle('M2 Max on AC power: synchronized throughput, process memory, and driver snapshots', fontsize=17, y=.98)
fig.subplots_adjust(top=.82, bottom=.31, left=.06, right=.98, wspace=.3)
save(fig, 'mac_benchmark',
     '1,000 updates per mode; fixed real batch, same initial weights, synchronized metrics. Torch compiled throughput is 13.8% below JAX.\n'
     'Torch end MPS tensors / driver allocation: eager 55.6 / 138.7 MiB; compiled 55.6 / 82.6 MiB. JAX GPU allocation was not measured.\n'
     'Unified-memory RSS and MPS counters overlap: do not sum them. Proxy snapshots include GUI/background work, not kernel occupancy.\n'
     'One run per mode in a fixed order; no frequency/temperature lock. This fixed-batch memory excludes the full offline training dataset.')

manifest = json.loads((ROOT / 'evidence/showcase_manifest.json').read_text())
fig, axes = plt.subplots(2, 4, figsize=(14, 8))
titles = ['State seed 0 / 500k', 'State seed 0 / 1M', 'Visual seed 613 / 40k\nStandard starts', 'Visual seed 613 / 40k\nXY displacement', '60k control / normal', '60k augmented / normal', '60k control / dim ×0.7', '60k augmented / dim ×0.7']
for ax, item, title in zip(axes.flat, manifest['videos'], titles):
    with imageio.get_reader(ROOT / item['path']) as stream:
        picture = stream.get_data(item['video_metadata']['thumbnail_frame'])
    ax.imshow(picture, interpolation='nearest')
    row = item['episode_record']
    status = 'SUCCESS' if row['success'] == '1' else 'FAILURE'
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(f"{status} · {row['length']} environment steps", fontsize=11,
                  color=TEAL if row['success'] == '1' else '#AD4853')
    ax.set(xticks=[], yticks=[])
fig.suptitle('Saved simulation cases: successful and unsuccessful fixed episode-0 examples', fontsize=16, y=.98)
fig.subplots_adjust(top=.87, bottom=.13, hspace=.5, wspace=.18)
save(fig, 'fixed_video_examples',
     'Midpoint frames from existing predetermined episode-0 videos; outcome labels come from saved episode CSVs.\n'
     'Stages and evaluation folds differ. These examples illustrate behavior; full evaluation tables determine success rates.')
print(json.dumps({'rendered_figures': len(figures), 'training_updates': 0, 'environment_steps': 0}))
