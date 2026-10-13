#!/usr/bin/env python3
"""Post-freeze fixed-probe CPU/MPS comparison; no training or episodes."""
import hashlib
import json
from pathlib import Path
import sys
import traceback

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / '复现/fql/pytorch'))
from fql_torch.networks import ActorVectorField
from fql_torch.runtime import configure_device, synchronize


def main():
    output = HERE / 'independent_mps_probe_validation.json'
    if output.exists():
        raise FileExistsError(output)
    record = {'status': 'running', 'device': 'mps', 'reference_device': 'cpu',
              'atol': 1e-4, 'rtol': 1e-4, 'training_updates': 0,
              'new_environment_episodes': 0, 'benchmark': False,
              'post_freeze_processing_script': True,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'policies': {}}
    try:
        device = configure_device('mps', 4)
        for window in (5, 10):
            directory = HERE / 'build' / f'average{window}'
            path = directory / 'student_policy.pt'
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            marker = json.loads((directory / 'student_policy.json').read_text())
            if digest != marker['sha256']:
                raise ValueError('Policy SHA mismatch')
            payload = torch.load(path, map_location='cpu', weights_only=True)
            c = payload['source_config']['agent']
            actor = ActorVectorField(28, 5, c['actor_hidden_dims'], c['actor_layer_norm'], with_time=False)
            actor.load_state_dict(payload['student'], strict=True)
            actor.eval()
            probe = payload['reload_probe']
            with torch.inference_mode():
                cpu = actor(probe['observations'], probe['noises']).clamp(-1, 1)
                torch.testing.assert_close(cpu, probe['actions'], atol=1e-6, rtol=0)
                actor.to(device)
                mps = actor(probe['observations'].to(device), probe['noises'].to(device)).clamp(-1, 1)
                synchronize(device)
                actual = mps.cpu()
            torch.testing.assert_close(actual, cpu, atol=1e-4, rtol=1e-4)
            if not bool(torch.isfinite(actual).all()):
                raise ValueError('Nonfinite MPS actions')
            record['policies'][str(window)] = {'artifact_sha256': digest, 'probe_cases': 4,
                'shape': list(actual.shape), 'passed': True,
                'maximum_absolute_action_difference': float((actual-cpu).abs().max()),
                'maximum_absolute_cpu_reload_difference': float((cpu-probe['actions']).abs().max())}
            print(json.dumps({'window': window, **record['policies'][str(window)]}), flush=True)
            del actor, payload, mps, actual, cpu
        record['status'] = 'passed'
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    with output.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': record['status']}), flush=True)
    if record['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
