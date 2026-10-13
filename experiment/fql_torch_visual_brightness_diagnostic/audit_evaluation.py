"""CPU replay and paired-statistic audit of the twelve common-case confirmation groups."""
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
status=read(OUT/'evaluation_status.json');assert status['status']=='completed_pending_audit' and status['episodes']==600
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
paired={};case_rows=[];case_means={};sampled_suites={};total_net_loss=0
streams=np.random.SeedSequence(20261020).spawn(2)
for suite,es,az,stream in [('standard',134000,135000,streams[0]),('shifted',136000,137000,streams[1])]:
    reference=f'seed613_clean_{suite}'
    for seed in (613,614,615):
        for condition in ('clean','dim07'):
            name=f'seed{seed}_{condition}_{suite}'
            assert np.array_equal(initial[name],initial[reference])
            assert np.array_equal(rgb_initial[name],rgb_initial[reference])
    indices=np.random.default_rng(stream).integers(0,50,size=(10000,50))
    differences=[];per_seed={}
    for seed in (613,614,615):
        clean=outcomes[f'seed{seed}_clean_{suite}'];dim=outcomes[f'seed{seed}_dim07_{suite}']
        diff=clean-dim;differences.append(diff);total_net_loss+=int(diff.sum())
        sampled=diff[indices].mean(1)*100
        per_seed[str(seed)]={'loss_pp':float(diff.mean()*100),'paired95_pp':np.quantile(sampled,[.025,.975]).tolist(),
            'lost_successes':int(np.sum((clean==1)&(dim==0))),'gained_successes':int(np.sum((clean==0)&(dim==1)))}
    case_mean=np.stack(differences).mean(0);case_means[suite]=case_mean
    sampled_suites[suite]=case_mean[indices].mean(1)*100
    paired[suite]={'per_seed':per_seed,'fixed_three_seed_mean_loss_pp':float(case_mean.mean()*100),
        'paired_case_cluster95_pp':np.quantile(sampled_suites[suite],[.025,.975]).tolist(),
        'model_rate_mean':{condition:float(np.mean([outcomes[f'seed{s}_{condition}_{suite}'].mean() for s in (613,614,615)])*100)
                           for condition in ('clean','dim07')}}
    for ep in range(50):
        case_rows.append({'suite':suite,'episode':ep,'env_seed':es+ep,'action_seed':az+ep,
            **{f'seed{seed}_{condition}':int(outcomes[f'seed{seed}_{condition}_{suite}'][ep])
               for seed in (613,614,615) for condition in ('clean','dim07')}})
macro_sampled=(sampled_suites['standard']+sampled_suites['shifted'])/2
macro_loss=total_net_loss/3
macro_ci=np.quantile(macro_sampled,[.025,.975]).tolist()
gate={'macro_mean_loss_pp':macro_loss,'paired_stratified_case_cluster95_pp':macro_ci,
    'net_lost_seed_case_pairs':total_net_loss,'threshold_loss_pp':10,
    'threshold_positive95_lower_bound':True,'passed':bool(total_net_loss>=30 and macro_ci[0]>0),
    'definition':'standard/shifted equal-weight; three fixed models equal-weight; 300seed-case pairs nested in100case clusters',
    'scope':'conditional on three fixed trained models; not population training-seed confidence'}
with (OUT/'paired_cases.csv').open('x',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(case_rows[0]));writer.writeheader();writer.writerows(case_rows)
assert all(sha256(Path(p))==h for p,h in freeze['inputs_sha256'].items())
record={'status':'passed','scope':'Fixed full-window visual models, paired clean versus synthetic brightness0.7',
    'groups':groups,'paired':paired,'primary_gate':gate,'episodes':600,'unique_case_clusters':100,
    'environment_steps':sum(g['environment_steps'] for g in groups.values()),
    'all_models_and_conditions_initial_state_and_raw_RGB_match':True,'saved_pixel_transforms_exact':True,
    'all_noises_rewards_success_and_terminal_fields_audited':True,'frozen_inputs_unchanged':True,
    'audit_training_updates':0,'audit_added_environment_steps':0,'paired_cases_sha256':sha256(OUT/'paired_cases.csv'),
    'audit_source_sha256':sha256(Path(__file__)),
    'limitations':['synthetic policy-pixel brightness, not physical lighting/camera/realrobot evidence',
        'only ep0 and fixedprobes pixel/actions independently rederived; all trajectory noises/rewards/success/end audited',
        'conditional on fixed three models, seed613-collected data and learned seed0teacher; no language/addedFQLRL']}
with (OUT/'evaluation_validation.json').open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
print(json.dumps({'status':'passed','episodes':600,'primary_gate':gate,'paired':paired}),flush=True)
