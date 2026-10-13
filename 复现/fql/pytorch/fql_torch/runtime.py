"""Explicit device selection, evidence and bounded host synchronization."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import torch
import psutil

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def write_json(path, value, exclusive=False):
    path = Path(path)
    if exclusive and path.exists():
        raise FileExistsError(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def configure_device(name='mps', cpu_threads=4):
    if name not in ('cpu', 'mps'):
        raise ValueError('This verified runner supports cpu or mps only.')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('CPU fallback is forbidden; unset PYTORCH_ENABLE_MPS_FALLBACK.')
    if os.environ.get('PYTORCH_MPS_FAST_MATH') == '1':
        raise RuntimeError('The verified float32 baseline does not enable fast math.')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS was requested but is not available; no CPU fallback.')
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(cpu_threads)
    return torch.device(name)


def synchronize(device):
    if torch.device(device).type == 'mps':
        torch.mps.synchronize()


def memory(device='cpu'):
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = {'rss_bytes': psutil.Process().memory_info().rss,
           'peak_rss_bytes': int(peak if sys.platform == 'darwin' else peak * 1024)}
    if torch.device(device).type == 'mps':
        out.update(mps_allocated_bytes=torch.mps.current_allocated_memory(),
                   mps_driver_allocated_bytes=torch.mps.driver_allocated_memory())
        if hasattr(torch.mps, 'recommended_max_memory'):
            out['mps_recommended_max_bytes'] = torch.mps.recommended_max_memory()
    return out


def source_hashes():
    paths = sorted((BASE / 'fql_torch').glob('*.py'))
    return {str(p.relative_to(BASE)): sha256(p) for p in paths}


def runtime_record(device):
    names = ('torch', 'numpy', 'ogbench', 'mujoco', 'gymnasium', 'psutil', 'imageio', 'imageio-ffmpeg')
    packages = {name: importlib.metadata.version(name) for name in names}
    try:
        power = subprocess.check_output(['pmset', '-g', 'batt'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        power = None
    return {'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'machine': platform.machine(),
            'command': [sys.executable, *sys.argv], 'device': str(device), 'packages': packages,
            'mps_available': torch.backends.mps.is_available(), 'cpu_threads': torch.get_num_threads(),
            'flags': {k: os.environ.get(k) for k in ('PYTORCH_ENABLE_MPS_FALLBACK',
                'PYTORCH_MPS_FAST_MATH', 'PYTORCH_MPS_PREFER_METAL', 'PYTORCH_MPS_HIGH_WATERMARK_RATIO')},
            'source_sha256': source_hashes(), 'power_state': power,
            'memory': memory(device), 'dtype': 'float32', 'training_environment_steps': 0}
