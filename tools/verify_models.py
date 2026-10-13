#!/usr/bin/env python3
"""Check all portable model SHAs and action probes; zero environment interactions."""
import argparse
import json
from pathlib import Path
from portable_models import ROOT, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    names = json.loads((ROOT / 'models/manifest.json').read_text())['models']
    checks = [load_model(name, args.device)[2] for name in names]
    result = {'passed': all(check['passed'] for check in checks), 'models': checks,
              'environment_steps': 0, 'training_updates': 0, 'dataset_loaded': False,
              'teacher_loaded': False, 'resumable': False}
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
