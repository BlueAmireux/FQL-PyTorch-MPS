"""Independently reload12 checkpoints, rederive heldout labels and sampler boundaries."""
from pathlib import Path
import json
import sys
import numpy as np
import torch

OUT = Path(__file__).resolve().parent; ROOT = OUT.parents[1]
sys.path.insert(0,str(ROOT / '复现/fql/pytorch'))
from visual.common import load_student_checkpoint, load_teacher, sha256
torch.set_num_threads(4)
read = lambda p: json.loads(p.read_text())
protocol = read(OUT / 'protocol.json')
inputs = read(OUT / 'training_freeze.json')['inputs_sha256']
assert all(sha256(Path(p)) == h for p,h in inputs.items())
assert read(OUT / 'training_status.json')['status'] == 'completed_pending_audit'
teacher, identity = load_teacher('cpu')
validations = {}
for name,path,seed in [('old',Path(protocol['data']['old'])/'val.npz',614),
                       ('new',Path(protocol['data']['val'])/'data.npz',1617)]:
    with np.load(path,allow_pickle=False) as data:
        validations[name] = {key:torch.from_numpy(data[key].copy()) for key in ('images','proprio','states')}
    validations[name]['noises'] = torch.randn((len(validations[name]['images']),5),generator=torch.Generator().manual_seed(seed))

def metrics(model,data):
    raw,xyz = [],[]
    with torch.inference_mode():
        for start in range(0,len(data['images']),256):
            sl = slice(start,start+256)
            visual = model.encoder(data['images'][sl].float()/255)
            actual = model.actor(torch.cat((data['proprio'][sl],visual),-1),data['noises'][sl])
            target = teacher(data['states'][sl],data['noises'][sl])
            raw.append((actual-target).square())
            xyz.append((visual[:,:3]-data['states'][sl,19:22]).square())
    return {'raw_mse':torch.cat(raw).mean().item(),
            'xyz_euclidean_rmse_cm':(torch.cat(xyz).sum(1).mean().sqrt()*10).item()}

def tensors(value):
    if isinstance(value,torch.Tensor): yield value
    elif isinstance(value,dict):
        for v in value.values(): yield from tensors(v)
    elif isinstance(value,(tuple,list)):
        for v in value: yield from tensors(v)

records, accepted = {},[]
for declared in protocol['models']:
    seed = declared['seed']; parent_run = Path(declared['run'])
    _,parent = load_student_checkpoint(parent_run,40000,'cpu')
    old_rng,new_rng = map(np.random.default_rng,np.random.SeedSequence(617).spawn(2))
    old_rng.bit_generator.state = parent['sampler']['old']; new_rng.bit_generator.state = parent['sampler']['new']
    expected = {}
    for update in range(1,20001):
        old_rng.integers(0,5000,128); new_rng.integers(0,740,64); new_rng.integers(0,740,64)
        if update in (1000,20000): expected[40000+update] = {'old':old_rng.bit_generator.state,'new':new_rng.bit_generator.state}
    pairs,segments = {},{}
    for arm in ('control','augmentation'):
        name = f'seed{seed}_{arm}'; run = OUT/name
        segment = read(run/'segment_40000_60000.json'); config = read(run/'config.json')
        assert config['parent_checkpoint_sha256'] == declared['sha256'] and config['parent_step'] == 40000
        assert config['visual_training_seed'] == seed and config['brightness']['applied'] == (arm == 'augmentation')
        assert config['batch_size'] == 256 and config['learning_rate'] == 1e-4 and config['xyz_loss_weight'] == 1
        assert config['selected_new_pool'] == {'category':'student','behavior_id':0,'unique_samples':740,'episodes':20}
        assert segment['first_update']['all_gradients_finite'] and segment['first_update']['teacher_gradients_absent']
        assert segment['first_update']['encoder_max_abs_parameter_change'] > 0 and segment['teacher_parameters_unchanged']
        assert segment['start_step'] == 40000 and segment['additional_updates'] == 20000 and segment['completed_step'] == 60000
        assert segment['first_1000']['projected20k_seconds'] <= 1800
        rows = [json.loads(line) for line in (run/'training.jsonl').read_text().splitlines()]
        assert [r['update'] for r in rows] == [250,500,750,1000,*range(2000,20001,1000)]
        assert all(np.isfinite([r['loss'],r['action_mse'],r['xyz_mse']]).all() and r['all_parameters_finite'] and r['teacher_parameters_unchanged'] for r in rows)
        saved = {}
        for step in (41000,60000):
            model,payload = load_student_checkpoint(run,step,'cpu')
            assert payload['config'] == config and payload['teacher']['sha256'] == identity['sha256']
            assert payload['sampler'] == expected[step] and all(torch.isfinite(t).all() for t in tensors(payload))
            assert payload['optimizer']['param_groups'] == parent['optimizer']['param_groups']
            assert all(int(v['step'].item()) == step for v in payload['optimizer']['state'].values())
            for filename,digest in config['additional_source_sha256'].items():
                assert sha256(ROOT/'复现/fql/pytorch/visual'/filename) == digest
            saved[step] = payload
        actual = {key:metrics(model,data) for key,data in validations.items()}
        for key,value in actual.items():
            expected_val = segment['final_validation'][key]['normal']
            assert abs(value['raw_mse']-expected_val['raw_mse']) < 1e-5
            assert abs(value['xyz_euclidean_rmse_cm']-expected_val['xyz_euclidean_rmse_cm']) < 1e-3
        record = {'seed':seed,'arm':arm,'run':str(run),'step':60000,'sha256':sha256(run/'checkpoint_60000.pt'),
                  'CPU_reload_errors':{step:payload['reload_max_abs_error'] for step,payload in saved.items()},
                  'CPU_heldout_recomputation':actual,'first1000':segment['first_1000'],
                  'training_seconds':segment['training_seconds'],'memory':segment['memory'],
                  '12_saved_checkpoints_finite_and_Adam_steps_correct':True,'continued_parent_sampler_boundaries_match':True}
        records[name] = record; accepted.append({k:record[k] for k in ('seed','arm','run','step','sha256')})
        pairs[arm],segments[arm] = saved,segment
        print(json.dumps({'accepted':name,'CPU_reload_error':saved[60000]['reload_max_abs_error']}),flush=True)
    for step in (41000,60000):
        for key in ('cpu','mps'): assert torch.equal(pairs['control'][step]['rng'][key],pairs['augmentation'][step]['rng'][key])
        assert pairs['control'][step]['sampler'] == pairs['augmentation'][step]['sampler']
    assert segments['control']['initial_validation'] == segments['augmentation']['initial_validation']
    assert segments['control']['first_update']['brightness_factors_sha256'] == segments['augmentation']['first_update']['brightness_factors_sha256']
    assert segments['control']['brightness_factor_probes_sha256'] == segments['augmentation']['brightness_factor_probes_sha256']
    assert any(not torch.equal(v,pairs['augmentation'][60000]['model'][k]) for k,v in pairs['control'][60000]['model'].items())
assert all(sha256(Path(p)) == h for p,h in inputs.items())
report = {'status':'passed','models':accepted,'branches':records,'new_supervised_updates':120000,'new_FQL_RL_updates':0,
          'same_parent_Adam_initial_validation_and_data':True,'same_CPU_MPS_RNG_and_sampler_at41k60k':True,
          'six_brightness_factor_probe_batches_equal_within_seed':True,'frozen_inputs_unchanged':True,
          'audit_training_updates':0,'audit_environment_steps':0,'source_sha256':sha256(Path(__file__)),
          'limits':['checks saved states, record points,6 factor probes and CPU validation; not every update intermediate']}
with (OUT/'training_acceptance.json').open('x') as f: json.dump(report,f,indent=2); f.write('\n')
