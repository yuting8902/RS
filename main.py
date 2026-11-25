#!/usr/bin/env python3
"""Standalone class-wise representative normal-image selector. Run from project root."""
import argparse
import logging
import random
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from repselect.utils.config import load_configs
from repselect.data.datasets import discover_classes, load_label_manifest, build_class_records
from repselect.features.cache import FeatureCache
from repselect.runner.class_runner import run_class
from repselect.utils.io import atomic_json, save_rows


def main():
    p = argparse.ArgumentParser(description='Independent Class-wise Representative Normal Sample Selection')
    p.add_argument('--paths', default=str(ROOT / 'configs' / 'paths.yaml'))
    p.add_argument('--config', default=str(ROOT / 'configs' / 'selection.yaml'))
    p.add_argument('--classes', nargs='+', default=None)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--no-resume', action='store_true')
    p.add_argument('--log-level', default='INFO', choices=['DEBUG','INFO','WARNING'])
    args = p.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(asctime)s %(levelname)s: %(message)s')
    c = load_configs(args.paths, args.config)
    for k in ('root','cache_root'):
        v = Path(c['output'][k]).expanduser()
        c['output'][k] = str((ROOT / v).resolve() if not v.is_absolute() else v.resolve())
    c['selection']['resume'] = bool(c['selection'].get('resume', True)) and not args.no_resume
    random.seed(int(c['experiment']['seed']))
    np.random.seed(int(c['experiment']['seed']))
    classes = args.classes or discover_classes(c)
    labels = load_label_manifest(c['data'].get('label_manifest'))
    class_data = {}
    for cls in classes:
        groups = build_class_records(c, cls, labels)
        groups['candidate'] = [r for r in groups['candidate'] if r.label != 'confirmed_abnormal']
        if not groups['candidate']:
            raise ValueError(f'{cls}: no usable candidates after removing confirmed abnormalities')
        class_data[cls] = groups
        logging.info('class=%s initial=%d candidates=%d reference=%d confirmed_good=%d',
                     cls, len(groups['initial']), len(groups['candidate']), len(groups['reference']),
                     sum(r.label in ('verified_good','dataset_good') for r in groups['candidate']))
    if args.dry_run:
        for cls, groups in class_data.items():
            n = len(groups['candidate'])
            approx_cache_gib = n*n*2304*4/2**30
            approx_feat_gib = n*2304*1024*4/2**30
            print(f'{cls}: candidates={n}, estimated feature FP32={approx_feat_gib:.2f} GiB, '
                  f'coverage FP32 max-sim cache={approx_cache_gib:.2f} GiB '
                  '(assuming 2304 valid patches/image)')
        print('Dry run passed: no DINO model, PCA, or Coverage was executed.')
        return
    cache = FeatureCache(c)
    summaries = []
    for cls in classes:
        result = run_class(c, cls, features=cache, groups=class_data[cls])
        summaries.append({'class_name': cls, 'bootstrap': len(result['bootstrap']),
                          'accepted': len(result['accepted']), 'attempted': len(result['attempted']),
                          'final_pca_dim': result.get('final_pca_dim'),
                          'stop_reason': result['stop_reason'],
                          'runtime_seconds': result.get('runtime_seconds')})
    output = Path(c['output']['root']) / c['experiment']['run_name']
    save_rows(output / 'summary.csv', summaries)
    atomic_json(output / 'run_config.json', c)
    print(f'Completed {len(classes)} classes. Output: {output}')

if __name__ == '__main__':
    main()
