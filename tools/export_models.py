#!/usr/bin/env python3
"""Export the fixed final models from a private archive; no training or environments.

Output has weights, architecture, original SHA provenance and action probes only.
Absolute source paths, optimizer state, training RNG and data samplers are omitted.
"""
import argparse
import json
from pathlib import Path
import torch
from portable_models import ROOT, SCHEMA, construct, probe_actions, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True, help='Original private project archive.')
    args = parser.parse_args()
    origin = args.source_root.resolve()
    output = ROOT / 'models'
    if (output / 'manifest.json').exists():
        raise FileExistsError('Existing exports are immutable; use a new release directory.')
    output.mkdir(parents=True, exist_ok=True)
    choices = [('state_seed0_average5', 'experiment/fql_torch_policy_average_v2/build/average5/student_policy.pt')]
    choices += [(f'state_seed{seed}_average5', f'experiment/fql_torch_three_seed_average_compact/policies/seed{seed}_epoch1000000/student_policy.pt') for seed in (1, 2)]
    choices += [(f'visual_seed{seed}_{arm}', f'experiment/fql_torch_visual_brightness_augmentation/seed{seed}_{arm}/checkpoint_60000.pt') for seed in (613, 614, 615) for arm in ('control', 'augmentation')]
    manifest = {'schema': SCHEMA, 'inference_only': True, 'resumable': False,
                'scope': 'Nine fixed final inference exports; no new training or evaluation.', 'models': {}}
    torch.set_num_threads(4)
    for name, relative in choices:
        path = origin / relative
        marker_path = path.with_suffix('.json')
        marker = json.loads(marker_path.read_text())
        if sha256(path) != marker['sha256'] or path.stat().st_size != marker['bytes']:
            raise ValueError(f'Original source artifact/marker mismatch: {relative}')
        original = torch.load(path, map_location='cpu', weights_only=True)
        state_model = name.startswith('state_')
        if state_model:
            if (original['window'] != 5 or original['max_source_updates'] != 1000000
                    or original['source_epochs'] != [960000, 970000, 980000, 990000, 1000000]):
                raise ValueError('Expected the fixed final 1M five-snapshot average.')
            c = original['source_config']['agent']
            architecture = {'observation_dim': 28, 'action_dim': 5,
                            'hidden_dims': list(c['actor_hidden_dims']), 'layer_norm': c['actor_layer_norm']}
            sources = {f'复现/fql/pytorch/{key}': value for key, value in original['frozen_core_sha'].items()}
            weights, probe = original['student'], original['reload_probe']
            seed, updates = original['training_seed'], 1000000
            epochs = original['source_epochs']
        else:
            if original['schema'] != 'visual-distillation-v1' or original['step'] != 60000:
                raise ValueError('Expected the fixed final 60k visual student.')
            architecture = {'rgb_shape': [3, 64, 64], 'rgb_dtype': 'uint8', 'proprio_dim': 19,
                            'noise_dim': 5, 'latent_dim': 9, 'action_dim': 5,
                            'actor_hidden_dims': [512, 512, 512, 512]}
            sources = {f'复现/fql/pytorch/visual/{key}': value for key, value in original['source_sha256'].items()}
            sources.update({f'复现/fql/pytorch/visual/{key}': value for key, value in original['config']['additional_source_sha256'].items()})
            sources['复现/fql/pytorch/fql_torch/networks.py'] = sha256(origin / '复现/fql/pytorch/fql_torch/networks.py')
            weights, probe = original['model'], original['probe']
            seed, updates, epochs = original['config']['visual_training_seed'], 60000, None
        for file, digest in sources.items():
            if sha256(origin / file) != digest or sha256(ROOT / file) != digest:
                raise ValueError(f'Original or release frozen source changed: {file}')
        source = {'relative_path': relative, 'sha256': sha256(path),
                  'marker_sha256': sha256(marker_path), 'schema': original.get('schema'),
                  'training_seed': seed, 'completed_updates': updates,
                  'source_epochs': epochs, 'source_sha256': sources}
        payload = {'schema': SCHEMA, 'kind': 'state_average5' if state_model else 'visual_student',
                   'inference_only': True, 'resumable': False, 'architecture': architecture,
                   'source': source, 'state_dict': {k: v.detach().cpu().clone() for k, v in weights.items()},
                   'probe': {k: v.detach().cpu().clone() for k, v in probe.items()}}
        model = construct(payload).eval().requires_grad_(False)
        model.load_state_dict(payload['state_dict'], strict=True)
        with torch.inference_mode():
            actual = probe_actions(model, payload).cpu()
        tolerance = 1e-6 if state_model else 1e-4
        torch.testing.assert_close(actual, payload['probe']['actions'], atol=tolerance, rtol=0)
        target = output / f'{name}.pt'
        with target.open('xb') as stream:
            torch.save(payload, stream)
        manifest['models'][name] = {'file': target.name, 'kind': payload['kind'],
            'architecture': architecture, 'source': source, 'sha256': sha256(target),
            'bytes': target.stat().st_size,
            'export_cpu_probe_error': float((actual - payload['probe']['actions']).abs().max())}
        print(json.dumps({'model': name, 'bytes': target.stat().st_size, 'probe_passed': True}), flush=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
