"""Stage 2 Gate metrics: image-wise aggregation then mean over R_good."""
import numpy as np


def image_error_metric(errors, metric='mean', threshold=None):
    e = np.asarray(errors, dtype=np.float64)
    if metric == 'mean':
        return float(np.mean(e))
    if metric == 'p90':
        return float(np.percentile(e, 90))
    if metric == 'high_error_ratio':
        if threshold is None:
            raise ValueError('high_error_ratio requires fixed threshold')
        return float(np.mean(e > threshold))
    raise ValueError(f'Unknown residual metric: {metric}')


def reference_metrics(pca, records, feature_cache, gate_metric='mean', threshold=None):
    values = []
    p95 = []
    for rec in records:
        errors = pca.errors(feature_cache.get(rec))
        values.append(image_error_metric(errors, gate_metric, threshold))
        p95.append(float(np.percentile(errors, 95)))
    if not values:
        raise ValueError('R_good is empty')
    return {'loss': float(np.mean(values)), 'p95_mean': float(np.mean(p95)), 'per_image': values}


def calibrate_ratio_threshold(pca, records, feature_cache, quantile=0.95):
    errors = [pca.errors(feature_cache.get(rec)) for rec in records]
    return float(np.quantile(np.concatenate(errors), quantile))
