"""Load checked inference exports without datasets, teachers or optimizer state."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '复现/fql/pytorch'
sys.path.insert(0, str(SOURCE))
SCHEMA = 'fql-portable-inference-v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def construct(payload):
    from fql_torch.networks import ActorVectorField
    from visual.student import VisualStudent
    if payload['kind'] == 'state_average5':
        c = payload['architecture']
        return ActorVectorField(c['observation_dim'], c['action_dim'],
                                c['hidden_dims'], c['layer_norm'], with_time=False)
    if payload['kind'] == 'visual_student':
        return VisualStudent()
    raise ValueError('Unknown portable model kind.')


def probe_actions(model, payload, device='cpu'):
    probe = payload['probe']
    if payload['kind'] == 'state_average5':
        return model(probe['observations'].to(device), probe['noises'].to(device)).clamp(-1, 1)
    return model(*(probe[key].to(device) for key in ('images', 'proprio', 'noises')))


def load_model(name, device='cpu'):
    import torch
    from fql_torch.runtime import configure_device
    device = configure_device(device, 4)
    manifest = json.loads((ROOT / 'models/manifest.json').read_text())
    entry = manifest['models'][name]
    path = ROOT / 'models' / entry['file']
    if sha256(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
        raise ValueError('Portable model file differs from its manifest.')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload['schema'] != SCHEMA or payload['inference_only'] is not True
            or payload['resumable'] is not False
            or any(key in payload for key in ('optimizer', 'rng', 'sampler', 'config'))):
        raise ValueError('Expected an inference-only export; it cannot resume training.')
    if any(payload[key] != entry[key] for key in ('kind', 'architecture', 'source')):
        raise ValueError('Manifest and model provenance differ.')
    for relative, digest in payload['source']['source_sha256'].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f'Frozen implementation changed: {relative}')
    model = construct(payload).to(device).eval().requires_grad_(False)
    model.load_state_dict(payload['state_dict'], strict=True)
    with torch.inference_mode():
        actual = probe_actions(model, payload, device).cpu()
    expected = payload['probe']['actions']
    tolerance = 1e-4 if device.type == 'mps' or payload['kind'] == 'visual_student' else 1e-6
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0)
    return model, payload, {'passed': True, 'device': str(device), 'model': name,
                           'max_absolute_action_difference': float((actual - expected).abs().max()),
                           'absolute_tolerance': tolerance, 'sha256': entry['sha256'],
                           'bytes': entry['bytes'], 'inference_only': True, 'resumable': False}
