#!/usr/bin/env python3
"""Standalone PCA diagnostic: synthetic checks, plus optional real .npy features.

Examples:
  python scripts/test_pca.py
  python scripts/test_pca.py --features-npy /path/to/features.npy --fixed-dim 600 --device cuda
  python scripts/test_pca.py --features-npy /path/a.npy /path/b.npy --mode explained_variance --device cuda

Each real .npy file must be a [num_valid_patches, feature_dim] array (e.g. 2304 x 1024).
No real model, image loading, or existing SubspaceAD project import is required.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import numpy as np
from repselect.pca.model import Stats, image_stats, fit_from_stats


def synthetic_check():
    rng = np.random.default_rng(2026)
    x1 = rng.normal(size=(50, 7)).astype(np.float32)
    x2 = rng.normal(size=(70, 7)).astype(np.float32) + 0.25
    stats = image_stats(x1, device='cpu').plus(image_stats(x2, device='cpu'))
    full = np.concatenate([x1, x2])
    mu = np.mean(full, axis=0, dtype=np.float64)
    centered = full.astype(np.float64) - mu
    eigenvalues, eigenvectors = np.linalg.eigh((centered.T @ centered) / (len(full)-1))
    k = 3
    expected = np.sum((centered - centered @ eigenvectors[:, -k:] @ eigenvectors[:, -k:].T)**2, axis=1)
    config = {'mode': 'fixed', 'fixed_dim': k, 'eigen_device': 'cpu', 'whiten': False, 'drop_k': 0}
    pca = fit_from_stats(stats, config)
    np.testing.assert_allclose(pca.mu, mu, atol=1e-11)
    np.testing.assert_allclose(pca.errors(full, device='cpu'), expected, atol=1e-9, rtol=1e-9)
    assert pca.retained == k
    config['mode'] = 'explained_variance'
    config['explained_variance'] = .99
    pca_ev = fit_from_stats(stats, config)
    assert 1 <= pca_ev.retained <= 7
    print(f'PASS: synthetic PCA mean, additive statistics, reconstruction, fixed-d={k}, EV99 dim={pca_ev.retained}')


def real_feature_check(paths, mode, fixed_dim, explained_variance, device):
    stats = None
    arrays = []
    feature_dim = None
    for filename in paths:
        path = Path(filename).expanduser()
        arr = np.load(path, mmap_mode='r', allow_pickle=False)
        if arr.ndim != 2 or len(arr) == 0:
            raise ValueError(f'{path}: expected nonempty [patches, dim] array, got {arr.shape}')
        if feature_dim is None:
            feature_dim = arr.shape[1]
        elif feature_dim != arr.shape[1]:
            raise ValueError(f'{path}: feature dimension {arr.shape[1]} differs from {feature_dim}')
        if not np.isfinite(arr).all():
            raise ValueError(f'{path}: nonfinite values')
        piece = image_stats(arr, device=device)
        stats = piece if stats is None else stats.plus(piece)
        arrays.append((str(path), arr))
    config = {'mode': mode, 'fixed_dim': fixed_dim, 'explained_variance': explained_variance,
              'eigen_device': device, 'whiten': False, 'drop_k': 0}
    pca = fit_from_stats(stats, config)
    print(f'PASS: PCA fitted, observations={stats.n}, feature_dim={feature_dim}, retained={pca.retained}, mode={mode}')
    for name, arr in arrays:
        errors = pca.errors(arr, device=device)
        print(f'  {name}: patches={len(errors)} residual_mean={errors.mean():.8g} p90={np.percentile(errors,90):.8g} finite={np.isfinite(errors).all()}')
        assert np.isfinite(errors).all()
        assert (errors >= 0).all()
    print('NOTE: This verifies PCA internals only; numerical parity with the company three-stage PCA still requires an independent feature/PCA comparison on the server.')


def main():
    parser = argparse.ArgumentParser(description='Standalone PCA tests and optional real-feature diagnostic')
    parser.add_argument('--features-npy', nargs='+', default=[], help='Optional cached DINO patch feature .npy files')
    parser.add_argument('--mode', choices=['fixed', 'explained_variance'], default='fixed')
    parser.add_argument('--fixed-dim', type=int, default=600)
    parser.add_argument('--explained-variance', type=float, default=.99)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    synthetic_check()
    if args.features_npy:
        real_feature_check(args.features_npy, args.mode, args.fixed_dim, args.explained_variance, args.device)
    else:
        print('Optional real-feature check: --features-npy image1.npy image2.npy --fixed-dim 600 --device cuda')
    print('PCA diagnostics passed.')


if __name__ == '__main__':
    main()
