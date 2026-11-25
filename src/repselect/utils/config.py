"""Strict configuration loader; all paths stay outside the selection source tree."""
import copy
import hashlib
import json
from pathlib import Path
import yaml


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def load_configs(paths_file, selection_file):
    paths_file, selection_file = Path(paths_file).resolve(), Path(selection_file).resolve()
    with paths_file.open(encoding='utf-8') as f:
        paths = yaml.safe_load(f) or {}
    with selection_file.open(encoding='utf-8') as f:
        selection = yaml.safe_load(f) or {}
    c = _merge(copy.deepcopy(paths), selection)
    validate(c)
    return c


def validate(c):
    if c['feature']['variant'] != 'a1_adapted_final':
        raise ValueError('v1 supports ONLY the actual a1_adapted_final feature variant')
    if int(c['feature']['image_size']) != 672:
        raise ValueError('A1 parity requires image_size=672 in this v1 project')
    if c['stage1']['score'] != 'mean':
        raise NotImplementedError('Stage 1 Top-K/Tail-aware are reserved for a validated later version')
    if c['stage2']['gate_metric'] not in ('mean', 'p90', 'high_error_ratio'):
        raise ValueError('Unknown gate metric')
    if c['pca']['mode'] not in ('fixed', 'explained_variance'):
        raise ValueError('PCA mode must be fixed or explained_variance')
    if c['pca']['mode'] == 'fixed' and int(c['pca']['fixed_dim']) < 1:
        raise ValueError('fixed_dim must be >=1')
    if not 0 < float(c['pca']['explained_variance']) <= 1:
        raise ValueError('explained_variance must be in (0,1]')
    if c['augmentation']['enabled'] or int(c['augmentation']['count']) != 0:
        raise NotImplementedError('v1 deliberately disables augmentation; Stage 2 rotate is future work')
    if c['selection']['mode'] not in ('two_stage', 'coverage_only', 'random_only', 'random_gate', 'residual_only'):
        raise ValueError('Unknown selection.mode')
    if c['stage1']['cache_mode'] not in ('precompute', 'stream', 'auto'):
        raise ValueError('Unknown stage1.cache_mode')
    if c['bootstrap']['method'] not in ('random', 'coverage'):
        raise ValueError('bootstrap.method must be random or coverage')
    if c['stage1']['compute_dtype'] not in ('float16', 'float32'):
        raise ValueError('stage1.compute_dtype must be float16 or float32')
    if int(c['selection']['max_new_images_per_class']) < 0:
        raise ValueError('Budget must be nonnegative')


def fingerprint(c):
    # Run identity includes config and model paths/versions but not transient output root.
    canonical = copy.deepcopy(c)
    canonical.pop('output', None)
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, default=str).encode()).hexdigest()
