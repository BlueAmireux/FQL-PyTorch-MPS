#!/usr/bin/env python3
"""Post-freeze CPU proof of student averaging; no training or environment."""
import hashlib
import io
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / 'experiment/fql_torch_migration/random_mps1000'
PREFIX = 'actor_onestep_flow.'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def tensor_count(value, prefix='payload'):
    if isinstance(value, torch.Tensor):
        require(value.device.type == 'cpu', f'{prefix}: non-CPU saved tensor')
        require(bool(torch.isfinite(value).all()), f'{prefix}: nonfinite saved tensor')
        return 1
    if isinstance(value, dict):
        return sum(tensor_count(child, f'{prefix}/{name}') for name, child in value.items())
    if isinstance(value, (list, tuple)):
        return sum(tensor_count(child, f'{prefix}/{i}') for i, child in enumerate(value))
    return 0


def verify():
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    frozen = read_json(HERE / 'inputs_before.json')['sha256']
    reference = read_json(RUN / 'config.json')
    sums = {5: {}, 10: {}}
    source_records, shape_reference, probe_reference = [], None, None
    for epoch in range(910000, 1000001, 10000):
        path = RUN / f'checkpoint_{epoch}.pt'
        marker_path = path.with_suffix('.json')
        # Read each full source payload from disk exactly once.
        stored = path.read_bytes()
        digest = sha_bytes(stored)
        marker = read_json(marker_path)
        require(digest == marker['sha256'] == frozen[str(path.relative_to(ROOT))],
                f'{epoch}: source payload SHA')
        require(sha_bytes(marker_path.read_bytes()) == frozen[str(marker_path.relative_to(ROOT))],
                f'{epoch}: marker SHA')
        payload = torch.load(io.BytesIO(stored), map_location='cpu', weights_only=True)
        require(payload['completed_updates'] == marker['completed_updates'] == epoch,
                f'{epoch}: wrong source epoch')
        require(canonical(payload['config']) == canonical(reference), f'{epoch}: config mismatch')
        require(payload['source_sha256'] == marker['source_sha256'], f'{epoch}: core identity')
        count = tensor_count(payload)
        student = {key[len(PREFIX):]: value for key, value in payload['model'].items()
                   if key.startswith(PREFIX)}
        shapes = {key: tuple(value.shape) for key, value in student.items()}
        require(len(student) == 10 and all(key.startswith('mlp.layers.') for key in student),
                f'{epoch}: expected ten student-only weight/bias tensors')
        if shape_reference is None:
            shape_reference = shapes
        require(shapes == shape_reference, f'{epoch}: student architecture mismatch')
        for key, value in student.items():
            require(value.dtype == torch.float32, f'{epoch}/{key}: source dtype')
            array = value.numpy().astype(np.float64)
            for window in (5, 10):
                if window == 5 and epoch < 960000:
                    continue
                if key not in sums[window]:
                    sums[window][key] = array.copy()
                else:
                    sums[window][key] += array
        source_records.append({'epoch': epoch, 'checkpoint_sha256': digest,
                               'finite_saved_tensors': count})
        if epoch == 1000000:
            probe_reference = {key: payload['reload_probe'][key].numpy().copy()
                               for key in ('observations', 'noises')}
        del student, payload, stored

    import sys
    sys.path.insert(0, str(ROOT / '复现/fql/pytorch'))
    from fql_torch.networks import ActorVectorField
    artifacts = {}
    all_exact = True
    for window in (5, 10):
        directory = HERE / 'build' / f'average{window}'
        path = directory / 'student_policy.pt'
        marker = read_json(directory / 'student_policy.json')
        stored = path.read_bytes()
        digest = sha_bytes(stored)
        require(digest == marker['sha256'] and len(stored) == marker['bytes'],
                f'average{window}: artifact SHA/size')
        policy = torch.load(io.BytesIO(stored), map_location='cpu', weights_only=True)
        require(policy['schema'] == marker['schema'] == 'inference_only_student_average_v1',
                f'average{window}: schema')
        require(policy['inference_only'] is True and policy['resumable'] is False
                and marker['inference_only'] is True and marker['resumable'] is False,
                f'average{window}: inference-only scope')
        require(not {'optimizer', 'rng', 'sampler', 'model'}.intersection(policy),
                f'average{window}: unexpected training state')
        epochs = list(range(1000000-(window-1)*10000, 1000001, 10000))
        require(policy['source_epochs'] == marker['source_epochs'] == epochs
                and policy['window'] == marker['window'] == window
                and policy['weights'] == marker['weights'] == [1/window]*window
                and policy['additional_optimizer_updates'] == 0 and policy['max_source_updates'] == 1000000,
                f'average{window}: method/epoch/weight identity')
        require(canonical(policy['source_config']) == canonical(reference), f'average{window}: config')
        require(policy['generation_script_sha256'] == sha_bytes((HERE / 'build_policy.py').read_bytes()),
                f'average{window}: builder identity')
        expected_records = {item['epoch']: item['checkpoint_sha256'] for item in source_records}
        for item in policy['source_checkpoints']:
            require(item['epoch'] in epochs and item['sha256'] == expected_records[item['epoch']],
                    f'average{window}: source provenance')
        require([item['epoch'] for item in policy['source_checkpoints']] == epochs,
                f'average{window}: complete ordered provenance')
        finite = tensor_count(policy)
        require(set(policy['student']) == set(sums[window]), f'average{window}: student keys')
        comparisons = {}
        for key, total in sums[window].items():
            expected = np.divide(total, np.float64(window)).astype(np.float32)
            actual = policy['student'][key].numpy()
            require(actual.dtype == np.float32 and actual.shape == expected.shape,
                    f'average{window}/{key}: student shape/dtype')
            exact = bool(np.array_equal(actual, expected))
            bitwise = bool(np.array_equal(actual.view(np.uint32), expected.view(np.uint32)))
            difference = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
            comparisons[key] = {'shape': list(actual.shape), 'value_exact': exact,
                                'bitwise_exact': bitwise,
                                'mismatched_values': int(np.count_nonzero(actual != expected)),
                                'maximum_absolute_difference': float(difference.max())}
            all_exact = all_exact and exact
        for key, expected_shape in (('observations', (4, 28)), ('noises', (4, 5))):
            actual = policy['reload_probe'][key].numpy()
            require(actual.dtype == np.float32 and actual.shape == expected_shape,
                    f'average{window}: probe {key} dtype/shape')
            np.testing.assert_array_equal(actual, probe_reference[key])
        c = policy['source_config']['agent']
        actor = ActorVectorField(28, 5, c['actor_hidden_dims'], c['actor_layer_norm'], with_time=False)
        actor.load_state_dict(policy['student'], strict=True)
        actor.eval()
        with torch.inference_mode():
            action = actor(policy['reload_probe']['observations'], policy['reload_probe']['noises']).clamp(-1, 1)
        torch.testing.assert_close(action, policy['reload_probe']['actions'], atol=1e-6, rtol=0)
        error = float((action-policy['reload_probe']['actions']).abs().max())
        artifacts[str(window)] = {'sha256': digest, 'finite_saved_tensors': finite,
                                  'source_epochs': epochs, 'student_tensors': len(comparisons),
                                  'parameter_comparisons': comparisons,
                                  'maximum_absolute_parameter_difference': max(v['maximum_absolute_difference']
                                      for v in comparisons.values()),
                                  'all_parameter_values_exact': all(v['value_exact'] for v in comparisons.values()),
                                  'reload_probe_maximum_absolute_action_difference': error,
                                  'inference_only_without_optimizer_rng_sampler': True}
        print(json.dumps({'window': window, **{key: value for key, value in artifacts[str(window)].items()
              if key != 'parameter_comparisons'}}), flush=True)
        del actor, policy, stored
    return {'status': 'passed' if all_exact else 'numerical_mismatch', 'validator_device': 'cpu',
            'environment_created': False, 'training_updates': 0, 'new_evaluation_episodes': 0,
            'method': 'Independent NumPy float64 ascending sum/division then float32; all student tensors compared',
            'full_source_payload_disk_reads': 10, 'source_checkpoints': source_records,
            'finite_source_saved_tensors': sum(item['finite_saved_tensors'] for item in source_records),
            'artifacts': artifacts, 'all_parameter_values_exact': all_exact,
            'post_freeze_processing_script': True, 'validator_sha256': sha_bytes(Path(__file__).read_bytes())}


def main():
    path = HERE / 'independent_average_validation.json'
    require(not path.exists(), 'Existing independent validation is immutable')
    started = time.perf_counter()
    try:
        result = verify()
    except BaseException as error:
        result = {'status': 'failed', 'error': str(error), 'error_type': type(error).__name__,
                  'traceback': traceback.format_exc(), 'validator_sha256': sha_bytes(Path(__file__).read_bytes()),
                  'post_freeze_processing_script': True, 'training_updates': 0, 'environment_created': False}
    result['wall_seconds'] = time.perf_counter()-started
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'wall_seconds': result['wall_seconds']}), flush=True)
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
