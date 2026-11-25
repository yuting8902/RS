"""Centered covariance PCA compatible in mathematics with SubspaceAD PCAModel.
Supports additive per-image sufficient statistics for fast Sequential trial fits.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class Stats:
    n: int
    s: np.ndarray
    q: np.ndarray

    @classmethod
    def zeros(cls, d):
        return cls(0, np.zeros(d, np.float64), np.zeros((d, d), np.float64))

    def plus(self, other):
        return Stats(self.n + other.n, self.s + other.s, self.q + other.q)


def image_stats(raw_features, device='auto', chunk=4096):
    import torch
    x = np.asarray(raw_features)
    if x.ndim != 2 or not x.shape[0] or not np.isfinite(x).all():
        raise ValueError('PCA requires finite nonempty [P,D] features')
    target = 'cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device)
    s = torch.zeros(x.shape[1], dtype=torch.float64, device=target)
    q = torch.zeros((x.shape[1], x.shape[1]), dtype=torch.float64, device=target)
    with torch.inference_mode():
        for st in range(0, x.shape[0], chunk):
            b = torch.as_tensor(np.asarray(x[st:st+chunk]).copy(), dtype=torch.float64, device=target)
            s += b.sum(0)
            q += b.T @ b
    return Stats(x.shape[0], s.cpu().numpy(), q.cpu().numpy())

@dataclass
class PCA:
    mu: np.ndarray
    components: np.ndarray
    eigvals: np.ndarray
    retained: int

    def errors(self, features, chunk=4096, device='auto'):
        """Squared orthogonal reconstruction residual, float64 for Gate stability.

        Uses CUDA FP64 on H100 if available; the CPU path is mathematically the same.
        Model components are kept on the PCA object between reference images.
        """
        import torch
        arr = np.asarray(features)
        use_gpu = torch.cuda.is_available() if device == 'auto' else device == 'cuda'
        result = np.empty(arr.shape[0], dtype=np.float64)
        if use_gpu:
            cache = getattr(self, '_gpu_params', None)
            if cache is None:
                mu_gpu = torch.from_numpy(self.mu).to('cuda', dtype=torch.float64)
                w_gpu = torch.from_numpy(self.components).to('cuda', dtype=torch.float64)
                self._gpu_params = (mu_gpu, w_gpu)
            else:
                mu_gpu, w_gpu = cache
            with torch.inference_mode():
                for st in range(0, len(arr), chunk):
                    z = torch.as_tensor(np.asarray(arr[st:st+chunk]).copy(), dtype=torch.float64, device='cuda') - mu_gpu
                    projected = z @ w_gpu
                    e = (z*z).sum(dim=1) - (projected*projected).sum(dim=1)
                    result[st:st+chunk] = e.clamp(min=0).cpu().numpy()
        else:
            for st in range(0, len(arr), chunk):
                z = np.asarray(arr[st:st+chunk], dtype=np.float64) - self.mu
                projected = z @ self.components
                e = (z*z).sum(axis=1) - (projected*projected).sum(axis=1)
                result[st:st+chunk] = np.maximum(e, 0)
        return result


def fit_from_stats(stats, cfg):
    import torch
    if stats.n < 2:
        raise ValueError('PCA needs at least two valid patch observations')
    mu = stats.s / stats.n
    covariance = (stats.q - np.outer(stats.s, stats.s) / stats.n) / (stats.n - 1)
    covariance = (covariance + covariance.T) / 2
    device = cfg.get('eigen_device', 'auto')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    t = torch.from_numpy(covariance.copy()).to(device)
    with torch.inference_mode():
        values, vectors = torch.linalg.eigh(t)
        values = torch.flip(values, (0,)).clamp(min=0)
        vectors = torch.flip(vectors, (1,))
    vals = values.cpu().numpy()
    vectors = vectors.cpu().numpy()
    tol = max(1e-12, float(vals[0]) * 1e-11)
    effective_rank = int(np.count_nonzero(vals > tol))
    if effective_rank < 1:
        raise ValueError('PCA covariance has zero effective rank')
    if cfg['mode'] == 'fixed':
        k = int(cfg['fixed_dim'])
        if k > effective_rank:
            raise ValueError(f'Fixed PCA k={k} exceeds effective rank {effective_rank}; adjust fixed_dim or add credible bootstrap images')
    else:
        total = vals.sum()
        k = int(np.searchsorted(np.cumsum(vals)/total, float(cfg['explained_variance'])) + 1)
        k = min(k, effective_rank)
    if cfg.get('whiten', False) or int(cfg.get('drop_k', 0)):
        raise NotImplementedError('v1 supports the actual unwhitened, drop_k=0 PCA configuration only')
    return PCA(mu, vectors[:, :k].copy(), vals[:k].copy(), k)


class StatsRepository:
    """Compute/cached per-image stats only on demand; never allocate stats for 5k images."""
    def __init__(self, feature_cache, records_by_id, device='auto'):
        self.features = feature_cache
        self.records = records_by_id
        self.device = device
        self.memo = {}

    def get(self, image_id):
        if image_id not in self.memo:
            self.memo[image_id] = image_stats(self.features.get(self.records[image_id]), self.device)
        return self.memo[image_id]

    def combine(self, ids, feature_dim=None):
        ids = list(ids)
        if not ids:
            raise ValueError('Cannot build PCA statistics from empty image selection')
        if feature_dim is None:
            feature_dim = len(self.get(ids[0]).s)
        s = Stats.zeros(feature_dim)
        for i in ids:
            s = s.plus(self.get(i))
        return s
