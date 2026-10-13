"""CPU replay and paired-statistic audit of the six temporal-window groups."""
from pathlib import Path
import csv
import json
import math
import sys
import numpy as np
import torch

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'复现/fql/pytorch'))
from visual.common import load_student_checkpoint,sha256
torch.set_num_threads(4)
read=lambda p:json.loads(p.read_text())
freeze=read(OUT/'evaluation_freeze.json');assert all(sha256(Path(p))==h for p,h in freeze['inputs_sha256'].items())
status=read(OUT/'evaluation_status.json');assert status['status']=='completed_pending_audit' and status['episodes']==180
groups,initial,rgb_initial,outcomes={},{},{},{}
for item in read(OUT/'evaluation_commands.json'):
    name=item['name'];path=Path(item['output']);summary=read(path/'summary.json')
    data=np.load(path/'replay.npz',allow_pickle=False)
    with (path/'episodes.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    assert summary['status']=='completed' and summary['episodes']==len(rows)==30
    assert summary['replay_sha256']==sha256(path/'replay.npz')
    assert summary['device']=='cpu' and summary['optimizer_updates']==0 and not summary['offline_dataset_loaded']
    assert all(np.isfinite(data[k]).all() for k in ['states','terminal_states','noises','actions','rewards'])
    es,az=item['env_seed_start'],item['action_seed_start'];n=30
    assert np.array_equal(data['env_seeds'],np.arange(es,es+n)) and np.array_equal(data['action_seeds'],np.arange(az,az+n))
    argv=item['argv'];run=Path(argv[argv.index('--run')+1]);model,payload=load_student_checkpoint(run,40000,'cpu')
    assert payload['config']==read(run/'config.json') and summary['checkpoint_sha256']==sha256(run/'checkpoint_40000.pt')
    for filename,digest in payload['config']['additional_source_sha256'].items():assert sha256(ROOT/'复现/fql/pytorch/visual'/filename)==digest
    offsets=data['episode_offsets'];results=[];peaks=[]
    for ep,row in enumerate(rows):
        lo,hi=map(int,offsets[ep:ep+2]);length=hi-lo;generator=torch.Generator().manual_seed(az+ep)
        noise=torch.cat([torch.randn((1,5),generator=generator) for _ in range(length)]).numpy()
        assert np.array_equal(noise,data['noises'][lo:hi])
        post=np.concatenate([data['states'][lo+1:hi],data['terminal_states'][ep:ep+1]])
        distance=np.linalg.norm(post[:,19:22]/10+np.array([.425,0,0])-[.5,0,.02],axis=1)
        success=distance<=.04;assert np.array_equal(success,data['successes'][lo:hi])
        assert np.array_equal(success.astype(float)-1,data['rewards'][lo:hi])
        assert bool(success[-1])==bool(int(row['success']))==bool(int(row['ever_success']))
        assert int(row['length'])==length and float(row['return'])==data['rewards'][lo:hi].sum()
        assert int(row['env_seed'])==es+ep and int(row['action_seed'])==az+ep
        assert not np.any(data['terminated'][lo:hi-1]|data['truncated'][lo:hi-1])
        assert (row['terminated']=='True')==bool(data['terminated'][hi-1])==bool(success[-1])
        assert (row['truncated']=='True')==bool(data['truncated'][hi-1])
        assert length<=200 and (data['terminated'][hi-1] or data['truncated'][hi-1])
        results.append(int(success[-1]));peaks.append(float(max(data['states'][lo:hi,21].max(),data['terminal_states'][ep,21])/10))
    assert sum(results)==summary['successes'] and int(offsets[-1])==summary['evaluation_environment_steps']
    fraction=sum(results)/n;z=1.959963984540054;den=1+z*z/n;center=(fraction+z*z/(2*n))/den
    radius=z*math.sqrt(fraction*(1-fraction)/n+z*z/(4*n*n))/den;ci=[max(0,center-radius),min(1,center+radius)]
    assert np.allclose(ci,summary['success_rate_wilson_95_percent_ci'],atol=1e-12,rtol=0)
    probes=data['probe_action_indices'].astype(int);lookup={int(i):j for j,i in enumerate(probes)}
    initial[name]=data['states'][offsets[:-1]].copy();rgb_initial[name]=data['image_probes'][[lookup[int(i)] for i in offsets[:-1]]].copy()
    indices=np.concatenate([np.arange(int(offsets[1])),probes])
    images=np.concatenate([data['episode_0_images'][:-1].transpose(0,3,1,2).copy(),data['image_probes']])
    error=0.
    with torch.inference_mode():
        for start in range(0,len(indices),256):
            batch=indices[start:start+256]
            action=model(torch.from_numpy(images[start:start+256].copy()),torch.from_numpy(data['states'][batch,:19].copy()),torch.from_numpy(data['noises'][batch].copy()))
            assert torch.isfinite(action).all();error=max(error,float(np.abs(action.numpy()-data['actions'][batch]).max()))
        pose=model.encoder(torch.from_numpy(data['image_probes']).float()/255)[:,:3].numpy()
    assert error<=1e-4
    pose_error=float(np.sqrt(np.mean(np.sum(((pose-data['states'][probes,19:22])/10)**2,axis=1)))*100)
    if item['suite']=='shifted':
        assert len(summary['initializations'])==n
        for ep,record in enumerate(summary['initializations']):
            shift=np.random.default_rng(es+ep+1000000).uniform([-.03,-.06],[.03,.06])
            assert np.array_equal(shift,[record['dx'],record['dy']])
            assert record['initial_goal_distance_m']>.06 and record['robot_goal_ctrl_unchanged'] and record['initial_contact']==0
    outcomes[name]=np.array(results)
    groups[name]={'successes':sum(results),'episodes':n,'success_rate':fraction,'wilson95':ci,
                  'environment_steps':summary['evaluation_environment_steps'],'mean_length':summary['mean_length'],
                  'CPU_reload_error':payload['reload_max_abs_error'],'CPU_actions_audited':len(indices),'CPU_action_max_abs_error':error,
                  'live_xyz_probe_rmse_cm':pose_error,'live_probe_count':len(probes),
                  'failed_cube_peak_at_most5cm':sum(peak<=.05 for peak,s in zip(peaks,results) if not s),
                  'source_summary_sha256':sha256(path/'summary.json'),'replay_sha256':summary['replay_sha256']}
    print(json.dumps({'audited':name,'successes':sum(results),'CPU_action_error':error}),flush=True)
contrasts=[('student_full','student_prefix'),('student_full','teacher_prefix')]
paired={};case_rows=[]
for suite,es,az in [('standard',118000,119000),('shifted',120000,121000)]:
    reference=f'teacher_prefix_{suite}'
    for arm in ['student_full','student_prefix']:
        assert np.array_equal(initial[f'{arm}_{suite}'],initial[reference])
        assert np.array_equal(rgb_initial[f'{arm}_{suite}'],rgb_initial[reference])
    paired[suite]={}
    for a,b in contrasts:
        left,right=outcomes[f'{a}_{suite}'],outcomes[f'{b}_{suite}'];difference=left-right
        sampled=difference[np.random.default_rng(20261016).integers(0,30,size=(10000,30))].mean(1)*100
        paired[suite][f'{a}_minus_{b}']={'difference_pp':float(difference.mean()*100),'bootstrap95_pp':np.quantile(sampled,[.025,.975]).tolist(),
                                           'left_only_successes':int(np.sum((left==1)&(right==0))),'right_only_successes':int(np.sum((left==0)&(right==1)))}
    for ep in range(30):
        case_rows.append({'suite':suite,'episode':ep,'env_seed':es+ep,'action_seed':az+ep,
                          **{arm:int(outcomes[f'{arm}_{suite}'][ep]) for arm in ['student_full','student_prefix','teacher_prefix']}})
with (OUT/'paired_cases.csv').open('x',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(case_rows[0]));writer.writeheader();writer.writerows(case_rows)
assert all(sha256(Path(p))==h for p,h in freeze['inputs_sha256'].items())
record={'status':'passed','scope':'Development-only fixed-case fixed-frame student sampling-window intervention',
        'groups':groups,'paired':paired,'episodes':180,'environment_steps':sum(g['environment_steps'] for g in groups.values()),
        'all_models_initial_state_and_RGB_match':True,'frozen_inputs_unchanged':True,'audit_added_environment_steps':0,
        'paired_cases_sha256':sha256(OUT/'paired_cases.csv'),'audit_source_sha256':sha256(Path(__file__)),
        'limitations':['one visual seed/shared initialization','same counts and initial cases; whole-window sampling changes early/late density; Ni retains stopping-time conditioning',
                       'newarm/student-only targets still come from frozen teacher','onlynormal ep0+fixedprobes actions rederived','new development cases, not additional independent confirmation',
                       'within collected shift range; no language/other-task/realrobot inference','bootstrapcase variability is not trainingseed variability']}
with (OUT/'evaluation_validation.json').open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
print(json.dumps({'status':'passed','paired':paired,'episodes':180}))
