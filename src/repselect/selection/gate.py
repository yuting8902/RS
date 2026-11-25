"""Trial PCA is never committed here. Only controller can accept it."""
from dataclasses import dataclass
from ..data.datasets import GOOD_LABELS
from ..pca.model import fit_from_stats
from ..pca.residual import reference_metrics

@dataclass
class GateResult:
    status: str
    reason: str
    delta: float | None = None
    before: float | None = None
    after: float | None = None
    p95_before: float | None = None
    p95_after: float | None = None
    trial_model: object = None
    trial_stats: object = None

class PCAImprovementGate:
    def __init__(self, c, feature_cache, stats_repo, reference_records, threshold=None):
        self.c = c
        self.features = feature_cache
        self.repo = stats_repo
        self.refs = reference_records
        self.threshold = threshold

    def evaluate(self, record, current_pca, current_stats, baseline=None):
        if record.label not in GOOD_LABELS:
            return GateResult('pending', 'label_unverified' if record.label == 'unknown' else 'confirmed_abnormal')
        if len(self.refs) < int(self.c['stage2'].get('min_reference_images', 1)):
            return GateResult('pending', 'reference_good_insufficient')
        trial_stats = current_stats.plus(self.repo.get(record.id))
        try:
            trial_model = fit_from_stats(trial_stats, self.c['pca'])
            before = baseline or reference_metrics(current_pca, self.refs, self.features,
                self.c['stage2']['gate_metric'], self.threshold)
            after = reference_metrics(trial_model, self.refs, self.features,
                self.c['stage2']['gate_metric'], self.threshold)
        except (ValueError, RuntimeError) as exc:
            return GateResult('pending', f'pca_evaluation_failed:{str(exc)[:200]}')
        denominator = max(float(before['loss']), 1e-12)
        delta = (before['loss'] - after['loss']) / denominator
        max_p95_worsen = self.c['stage2'].get('reject_if_worse_p95_fraction')
        p95_bad = max_p95_worsen is not None and after['p95_mean'] > before['p95_mean'] * (1 + max_p95_worsen)
        accepted = delta > float(self.c['stage2']['delta_min']) and not p95_bad
        return GateResult('accept' if accepted else 'reject',
            'positive_pca_gain' if accepted else ('p95_worsened' if p95_bad else 'insufficient_gain'),
            float(delta), float(before['loss']), float(after['loss']),
            float(before['p95_mean']), float(after['p95_mean']), trial_model if accepted else None,
            trial_stats if accepted else None)
