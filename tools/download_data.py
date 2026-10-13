#!/usr/bin/env python3
"""Download the official state dataset and verify its pinned identity."""
import argparse
import hashlib
from pathlib import Path
import ogbench

EXPECTED = {
    'cube-single-play-v0.npz': '80f3b6fd27f4f9d9e9eb6f0d07d6951559012f45b1e15ea4046ef8ecd8d3684e',
    'cube-single-play-v0-val.npz': '96d07401bdebdc3f0ea6d56ed1333863e0962f441483adc2c43b83105046eb00',
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('DataSet/fql'))
    args = parser.parse_args()
    ogbench.download_datasets(['cube-single-play-v0'], dataset_dir=str(args.data_dir))
    for name, expected in EXPECTED.items():
        digest = hashlib.sha256()
        with (args.data_dir / name).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f'Official dataset SHA mismatch: {name}')
        print(f'verified {name}: {expected}')

if __name__ == '__main__':
    main()
