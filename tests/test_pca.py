"""Focused PCA numerical checks; no real model, checkpoint, or GPU required."""
import numpy as np
import pytest

from repselect.pca.model import Stats, image_stats, fit_from_stats
from repselect.pca.residual import image_error_metric, reference_metrics


def _cfg(mode='fixed', fixed_dim=3, explained_variance=0.99):
    return dict(mode=mode, fixed_dim=fixed_dim, explained_variance=explained_variance,
                eigen_device='cpu', whiten=False, drop_k=0)


def test_pca_fixed_dim_matches_numpy_reference():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(160, 8)).astype(np.float32)
    x += np.linspace(-1, 2, 8).astype(np.float32)
    model = fit_from_stats(image_stats(x, device='cpu'), _cfg(fixed_dim=3))
    center = x.astype(np.float64) - x.mean(axis=0, dtype=np.float64)
    cov = center.T @ center / (len(x) - 1)
    vals, vecs = np.linalg.eigh(cov)
    vals, vecs = vals[::-1], vecs[:, ::-1]
    expected = np.maximum(0, np.sum(center**2, axis=1) -
                          np.sum((center @ vecs[:, :3])**2, axis=1))
    np.testing.assert_allclose(model.mu, x.mean(axis=0, dtype=np.float64), atol=1e-12)
    np.testing.assert_allclose(model.eigvals, vals[:3], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(model.errors(x, device='cpu'), expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(model.components.T @ model.components, np.eye(3), atol=1e-10)


def test_pca_explained_variance_selects_minimum_dim():
    # Diagonal covariance with strictly ordered eigenvalues: 100, 10, 1, 0.01.
    x = np.vstack([np.eye(4), -np.eye(4)]).astype(np.float32)
    x *= np.array([10., np.sqrt(10.), 1., 0.1], dtype=np.float32)
    model = fit_from_stats(image_stats(x, device='cpu'), _cfg(mode='explained_variance', explained_variance=.99))
    values = np.linalg.eigvalsh(np.cov(x, rowvar=False))[::-1]
    expected_dim = int(np.searchsorted(np.cumsum(values) / values.sum(), .99) + 1)
    assert model.retained == expected_dim == 2


def test_pca_additive_sufficient_statistics_equal_direct_fit():
    rng = np.random.default_rng(432)
    images = [rng.normal(size=(20 + i * 5, 6)).astype(np.float32) for i in range(3)]
    stats = Stats.zeros(6)
    for img in images:
        stats = stats.plus(image_stats(img, device='cpu'))
    direct = image_stats(np.concatenate(images), device='cpu')
    np.testing.assert_allclose(stats.s, direct.s, atol=1e-11)
    np.testing.assert_allclose(stats.q, direct.q, atol=1e-11)
    fitted = fit_from_stats(stats, _cfg(fixed_dim=2))
    direct_fit = fit_from_stats(direct, _cfg(fixed_dim=2))
    np.testing.assert_allclose(fitted.errors(np.concatenate(images), device='cpu'),
                               direct_fit.errors(np.concatenate(images), device='cpu'), atol=1e-10)


def test_pca_reconstruction_error_is_nonnegative_and_reduced_with_more_components():
    rng = np.random.default_rng(456)
    x = rng.normal(size=(120, 7)).astype(np.float32)
    low = fit_from_stats(image_stats(x, device='cpu'), _cfg(fixed_dim=2))
    high = fit_from_stats(image_stats(x, device='cpu'), _cfg(fixed_dim=5))
    err_lo = low.errors(x, device='cpu')
    err_hi = high.errors(x, device='cpu')
    assert np.all(err_lo >= 0) and np.all(err_hi >= 0)
    assert np.all(err_hi <= err_lo + 1e-8)


def test_reference_aggregation_is_mean_of_per_image_metrics():
    class Record:
        def __init__(self, name): self.id = name
    class Cache:
        def __init__(self, arrays): self.arrays = arrays
        def get(self, record): return self.arrays[record.id]
    rng = np.random.default_rng(789)
    train = rng.normal(size=(40, 5)).astype(np.float32)
    pca = fit_from_stats(image_stats(train, device='cpu'), _cfg(fixed_dim=2))
    records = [Record('a'), Record('b')]
    cache = Cache({'a': rng.normal(size=(3, 5)).astype(np.float32),
                   'b': rng.normal(size=(11, 5)).astype(np.float32)})
    observed = reference_metrics(pca, records, cache, gate_metric='p90')
    expected = np.mean([image_error_metric(pca.errors(cache.get(r), device='cpu'), 'p90') for r in records])
    assert observed['loss'] == pytest.approx(expected)
    assert len(observed['per_image']) == 2


def test_pca_rejects_dimension_beyond_effective_rank():
    x = np.tile(np.array([[0., 0., 0.], [1., 0., 0.]], dtype=np.float32), (5, 1))
    with pytest.raises(ValueError, match='effective rank'):
        fit_from_stats(image_stats(x, device='cpu'), _cfg(fixed_dim=2))
