"""CPU replay and paired-statistic audit of the 36 brightness-augmentation confirmation groups."""
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
status=read(OUT/'evaluation_status.json');assert status['status']=='completed_pending_audit' and status['episodes']==1800
groups,initial,rgb_initial,outcomes={},{},{},{}
for item in read(OUT/'evaluation_commands.json'):
    name=item['name'];path=Path(item['output']);summary=read(path/'summary.json')
    assert summary['brightness_factor']==item['brightness']
    assert summary['input_transform_sha256']==sha256(path/'input_transform.npz')
    data=np.load(path/'replay.npz',allow_pickle=False)
    with (path/'episodes.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    assert summary['status']=='completed' and summary['episodes']==len(rows)==50
    assert summary['replay_sha256']==sha256(path/'replay.npz')
    assert summary['device']=='cpu' and summary['optimizer_updates']==0 and not summary['offline_dataset_loaded']
    assert all(np.isfinite(data[k]).all() for k in ['states','terminal_states','noises','actions','rewards'])
    es,az=item['env_seed_start'],item['action_seed_start'];n=50
    assert np.array_equal(data['env_seeds'],np.arange(es,es+n)) and np.array_equal(data['action_seeds'],np.arange(az,az+n))
    argv=item['argv'];run=Path(argv[argv.index('--run')+1]);model,payload=load_student_checkpoint(run,60000,'cpu')
    assert payload['config']==read(run/'config.json') and summary['checkpoint_sha256']==sha256(run/'checkpoint_60000.pt')
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
    raw=np.load(path/'input_transform.npz',allow_pickle=False)
    assert np.array_equal(raw['probe_action_indices'],probes)
    transform=lambda rgb:np.clip(np.rint(rgb.astype(np.float32)*item['brightness']),0,255).astype(np.uint8)
    assert np.array_equal(transform(raw['raw_probe_images']),data['image_probes'])
    assert np.array_equal(transform(raw['raw_episode_0_images']),data['episode_0_images'])
    initial[name]=data['states'][offsets[:-1]].copy()
    rgb_initial[name]=raw['raw_probe_images'][[lookup[int(i)] for i in offsets[:-1]]].copy()
    indices=np.concatenate([np.arange(int(offsets[1])),probes])
    images=np.concatenate([data['episode_0_images'][:-1].transpose(0,3,1,2).copy(),data['image_probes']])
    error=0.
    with torch.inference_mode():
        for start in range(0,len(indices),256):
            batch=indices[start:start+256]
            action=model(torch.from_numpy(images[start:start+256].copy()),torch.from_numpy(data['states'][batch,:19].copy()),torch.from_numpy(data['noises'][batch].copy()))
            assert torch.isfinite(action).all();error=max(error,float(np.abs(action.numpy()-data['actions'][batch]).max()))
        pose=model.encoder(torch.from_numpy(data['image_probes']).float()/255)[:,:3].numpy()
    first_probe=np.array([lookup[int(i)] for i in offsets[:-1]])
    initial_pose_rmse_cm=float(np.sqrt(np.mean(np.sum(((pose[first_probe]-data['states'][offsets[:-1],19:22])/10)**2,axis=1)))*100)
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
                  'shared_initial_RGB_xyz_RMSE_cm':initial_pose_rmse_cm,
                  'failed_cube_peak_at_most5cm':sum(peak<=.05 for peak,s in zip(peaks,results) if not s),
                  'source_summary_sha256':sha256(path/'summary.json'),'replay_sha256':summary['replay_sha256']}
    print(json.dumps({'audited':name,'successes':sum(results),'CPU_action_error':error}),flush=True)
paired={};case_rows=[];sampled_suites={};macro_net={condition:0 for condition in ('clean','dim07','dim055')}
protocol=read(OUT/'protocol.json');streams=np.random.SeedSequence(20261021).spawn(2)
for suite,stream in zip(('standard','shifted'),streams):
    cases=protocol['evaluation'][suite];es,az=cases['env_seed_start'],cases['action_seed_start']
    reference=f'seed613_control_clean_{suite}'
    for seed in (613,614,615):
        for arm in ('control','augmentation'):
            for condition in ('clean','dim07','dim055'):
                name=f'seed{seed}_{arm}_{condition}_{suite}'
                assert np.array_equal(initial[name],initial[reference]) and np.array_equal(rgb_initial[name],rgb_initial[reference])
    indices=np.random.default_rng(stream).integers(0,50,size=(10000,50));paired[suite]={};sampled_suites[suite]={}
    for condition in ('clean','dim07','dim055'):
        differences=[];per_seed={}
        for seed in (613,614,615):
            control=outcomes[f'seed{seed}_control_{condition}_{suite}'];aug=outcomes[f'seed{seed}_augmentation_{condition}_{suite}']
            diff=aug-control;differences.append(diff);macro_net[condition]+=int(diff.sum())
            sampled=diff[indices].mean(1)*100
            per_seed[str(seed)]={'difference_pp':float(diff.mean()*100),'paired95_pp':np.quantile(sampled,[.025,.975]).tolist(),
                'rescued':int(np.sum((control==0)&(aug==1))),'introduced_failures':int(np.sum((control==1)&(aug==0)))}
        mean=np.stack(differences).mean(0);sampled_suites[suite][condition]=mean[indices].mean(1)*100
        paired[suite][condition]={'per_seed':per_seed,'fixed_three_seed_mean_difference_pp':float(mean.mean()*100),
            'paired_case_cluster95_pp':np.quantile(sampled_suites[suite][condition],[.025,.975]).tolist(),
            'model_rate_mean_percent':{arm:float(np.mean([outcomes[f'seed{s}_{arm}_{condition}_{suite}'].mean() for s in (613,614,615)])*100)
                                      for arm in ('control','augmentation')}}
    for ep in range(50):
        case_rows.append({'suite':suite,'episode':ep,'env_seed':es+ep,'action_seed':az+ep,
            **{f'seed{seed}_{arm}_{condition}':int(outcomes[f'seed{seed}_{arm}_{condition}_{suite}'][ep])
               for seed in (613,614,615) for arm in ('control','augmentation') for condition in ('clean','dim07','dim055')}})
macro={condition:{'difference_pp':macro_net[condition]/3,
                  'paired_stratified_case_cluster95_pp':np.quantile((sampled_suites['standard'][condition]+sampled_suites['shifted'][condition])/2,[.025,.975]).tolist(),
                  'net_seed_case_successes':macro_net[condition]} for condition in ('clean','dim07','dim055')}
primary=macro['dim07'];clean=macro['clean']
decision={'primary_benefit_gate':bool(primary['difference_pp']>=10 and primary['paired_stratified_case_cluster95_pp'][0]>0),
          'clean_cost_gate':bool(clean['difference_pp']>=-5 and clean['paired_stratified_case_cluster95_pp'][0]>=-10),
          'thresholds_predeclared':True,'no_second_trial_or_model_selection':True}
decision['recommend_augmentation_for_this_scope']=decision['primary_benefit_gate'] and decision['clean_cost_gate']
with (OUT/'paired_cases.csv').open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(case_rows[0]));writer.writeheader();writer.writerows(case_rows)
assert all(sha256(Path(p))==h for p,h in freeze['inputs_sha256'].items())
record={'status':'passed','groups':groups,'paired':paired,'macro':macro,'decision':decision,
        'episodes':1800,'unique_case_clusters':100,'environment_steps':sum(g['environment_steps'] for g in groups.values()),
        'all_models_conditions_initial_state_and_rawRGB_match':True,'saved_pixel_transforms_exact':True,
        'all_noises_rewards_official_success_and_terminal_fields_audited':True,'frozen_inputs_unchanged':True,
        'audit_training_updates':0,'audit_environment_steps':0,'paired_cases_sha256':sha256(OUT/'paired_cases.csv'),
        'audit_source_sha256':sha256(Path(__file__)),
        'limits':['synthetic brightness input, not physical light/camera/realrobot','conditional fixed3model case CIs, not seed population confidence',
                  'pixel/actions recomputed for ep0 and fixed probes; no claim about every pixel in all1800 trajectories','no newFQLRL or language']}
with (OUT/'evaluation_validation.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps({'status':'passed','episodes':1800,'macro':macro,'decision':decision}),flush=True)
