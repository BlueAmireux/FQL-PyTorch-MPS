"""Compact final results, fixed-seed-rate plots and every changed case in the CSV."""
from pathlib import Path
import csv
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
audit = read(OUT/'evaluation_validation.json'); training = read(OUT/'training_acceptance.json')
status = read(OUT/'evaluation_status.json'); protocol = read(OUT/'protocol.json')
assert audit['status'] == training['status'] == 'passed'
assert all(sha(Path(p)) == h for p,h in read(OUT/'evaluation_freeze.json')['inputs_sha256'].items())
conditions = ('clean','dim07','dim055'); suites = ('standard','shifted')
result = {'status':'completed_single_brightness_experiment','training_seeds':[613,614,615],
          'fixed_parent_step':40000,'fixed_end_step':60000,'groups':audit['groups'],
          'paired':audit['paired'],'macro':audit['macro'],'decision':audit['decision'],
          'cost':{'new_supervised_updates':120000,'new_FQL_RL_updates':0,'new_training_collection_episodes':0,
                  'training_controller_seconds':read(OUT/'training_status.json')['controller_seconds'],
                  'sum_training_loop_seconds':sum(r['training_seconds'] for r in training['branches'].values()),
                  'evaluation_episodes':1800,'unique_case_clusters':100,'evaluation_environment_steps':audit['environment_steps'],
                  'evaluation_controller_seconds':status['controller_seconds'],'max_concurrent_cpu_evaluators':2},
          'shared_initial_image_pose_RMSE_cm':{name:g['shared_initial_RGB_xyz_RMSE_cm'] for name,g in audit['groups'].items()},
          'limits':protocol['limits'],'frozen_inputs':len(read(OUT/'evaluation_freeze.json')['inputs_sha256']),
          'all_raw_episodes_and_fixed_ep0_videos_retained':True,
          'fixed_models_case_intervals_not_seed_population_confidence':True}
with (OUT/'results.json').open('x') as f: json.dump(result,f,indent=2); f.write('\n')
plots = OUT/'plots'; plots.mkdir(exist_ok=False)
colors = {'control':'#8a99a6','augmentation':'#207baf'}
fig,axes = plt.subplots(1,2,figsize=(10,4.4),sharey=True)
for ax,suite in zip(axes,suites):
    x = np.arange(3)
    for arm,offset in [('control',-.18),('augmentation',.18)]:
        rates = np.array([[audit['groups'][f'seed{s}_{arm}_{c}_{suite}']['success_rate']*100 for s in (613,614,615)] for c in conditions])
        ax.bar(x+offset,rates.mean(1),width=.34,color=colors[arm],label=arm.capitalize())
        for i,seed in enumerate((613,614,615)):
            ax.scatter(x+offset+(i-1)*.045,rates[:,i],marker=('o','s','^')[i],s=22,edgecolor='white',linewidth=.4,color='#1a2833',zorder=3)
    ax.set_xticks(x,['Normal (1.0)','Dim (0.7)','Dim (0.55)'])
    ax.set_title(suite.capitalize()); ax.set_ylim(0,105); ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Success rate (%)'); axes[1].legend(loc='lower right',fontsize=8)
fig.suptitle('Brightness augmentation vs equal-budget continuation\n3 fixed seeds, 50 shared cases per suite; dots are individual seeds',fontsize=11)
fig.tight_layout(); fig.savefig(plots/'success_by_condition.png',dpi=180); plt.close(fig)
fig,ax = plt.subplots(figsize=(7,3.5))
means = [audit['macro'][c]['difference_pp'] for c in conditions]
intervals = np.array([audit['macro'][c]['paired_stratified_case_cluster95_pp'] for c in conditions]).T
ax.errorbar(means,np.arange(3),xerr=np.vstack((np.array(means)-intervals[0],intervals[1]-np.array(means))),fmt='o',capsize=5,color='#207baf')
ax.axvline(0,color='#8a99a6',linewidth=1); ax.set_yticks(range(3),['Normal cost','Primary dim0.7','Secondary dim0.55']); ax.invert_yaxis()
ax.set_xlabel('Augmentation minus control (percentage points)')
ax.set_title('Equal-weight suites and3 fixed models\n95% case-cluster intervals, conditional on these models',fontsize=10)
ax.spines[['top','right']].set_visible(False); ax.grid(axis='x',alpha=.18)
fig.tight_layout(); fig.savefig(plots/'paired_difference.png',dpi=180); plt.close(fig)
rows = [['suite','condition','seed','control_successes','augmentation_successes','episodes','difference_pp','CI95_low_pp','CI95_high_pp']]
for suite in suites:
    for c in conditions:
        for seed in (613,614,615):
            paired = audit['paired'][suite][c]['per_seed'][str(seed)]
            rows.append([suite,c,seed,audit['groups'][f'seed{seed}_control_{c}_{suite}']['successes'],
                         audit['groups'][f'seed{seed}_augmentation_{c}_{suite}']['successes'],50,
                         paired['difference_pp'],*paired['paired95_pp']])
with (OUT/'summary_table.csv').open('x',newline='') as f: csv.writer(f).writerows(rows)
recommend = audit['decision']['recommend_augmentation_for_this_scope']
interpretation = {'status':'bounded_experiment_closed','recommend_augmentation_for_declared_scope':recommend,
                  'primary_benefit_gate':audit['decision']['primary_benefit_gate'],'clean_cost_gate':audit['decision']['clean_cost_gate'],
                  'findings':audit['macro'],'no_second_tuning_trial':True,'no_language':True,
                  'allowed_claim':'A controlled visual-policy training adaptation for this synthetic brightness condition; standard image cost is reported.',
                  'not_supported':['new FQL RL algorithm or original paper visual score','physical illumination/new camera/realrobot generalization','VLA','100% true success guarantee','population training-seed confidence'],
                  'original_baseline_and_all_negative_results_preserved':True}
with (OUT/'interpretation.json').open('x') as f: json.dump(interpretation,f,indent=2); f.write('\n')
print(json.dumps({'completed':True,'macro':audit['macro'],'decision':audit['decision'],'cost':result['cost']}),flush=True)
