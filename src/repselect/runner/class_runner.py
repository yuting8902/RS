"""Per-class transactional sequential controller; does not call SubspaceAD_."""
import hashlib
import json
import logging
import random
import time
from pathlib import Path

from ..data.datasets import GOOD_LABELS, build_class_records
from ..features.cache import FeatureCache
from ..pca.model import StatsRepository, fit_from_stats
from ..pca.residual import reference_metrics, image_error_metric, calibrate_ratio_threshold
from ..selection.coverage import CoverageEngine
from ..selection.gate import PCAImprovementGate
from ..utils.config import fingerprint
from ..utils.io import atomic_json, link_or_copy, save_rows


def _data_hash(groups):
    rows = [(split, r.id, r.path, r.digest, r.label, r.source)
            for split, arr in groups.items() for r in arr]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def _new_state(c, groups):
    return {'config_hash': fingerprint(c), 'data_hash': _data_hash(groups),
            'bootstrap': [], 'accepted': [], 'attempted': [], 'decisions': [],
            'ratio_threshold': None, 'residual_initial_scores': None,
            'stop_reason': None}


def _export_class(c, cls, groups, state, out):
    output = Path(c['output']['root']) / c['experiment']['run_name']
    mode = c['output'].get('image_export', 'copy')
    lookup = {r.id: r for arr in groups.values() for r in arr}
    chosen = state['bootstrap'] + state['accepted']
    for image_id in chosen:
        r = lookup[image_id]
        sub = Path(cls) / 'train' / 'good' / f'{r.digest[:10]}__{Path(r.path).name}'
        link_or_copy(r.path, output / 'selected_only' / sub, mode)
        link_or_copy(r.path, output / 'merged_training_set' / sub, mode)
    for r in groups['initial']:
        link_or_copy(r.path, output / 'merged_training_set' / cls / 'train' / 'good' / Path(r.path).name, mode)
    save_rows(out / 'selection_decisions.csv', state['decisions'])
    save_rows(out / 'selected_ids.csv', [{'image_id': i, 'phase': 'bootstrap' if i in state['bootstrap'] else 'gate'} for i in chosen])
    atomic_json(out / 'summary.json', {
        'class_name': cls, 'initial_count': len(groups['initial']),
        'candidate_count': len(groups['candidate']), 'reference_count': len(groups['reference']),
        'bootstrap_count': len(state['bootstrap']), 'accepted_count': len(state['accepted']),
        'total_new': len(chosen), 'attempted': len(state['attempted']),
        'stop_reason': state['stop_reason'], 'final_pca_dim': state.get('final_pca_dim'),
        'runtime_seconds': state.get('runtime_seconds'), 'ratio_threshold': state['ratio_threshold'],
    })


def run_class(c, cls, features=None, groups=None):
    start_time = time.monotonic()
    groups = groups if groups is not None else build_class_records(c, cls)
    # When called directly, maintain the same known-abnormal exclusion as main.
    groups['candidate'] = [r for r in groups['candidate'] if r.label != 'confirmed_abnormal']
    features = features if features is not None else FeatureCache(c)
    out = Path(c['output']['root']) / c['experiment']['run_name'] / 'logs' / cls
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / 'controller_state.json'
    if checkpoint.exists() and not c['selection'].get('resume', True):
        raise FileExistsError(
            f'{cls}: existing checkpoint at {checkpoint}. Change experiment.run_name '
            'for a fresh run, or enable Resume; this protects previous outputs.')
    if checkpoint.exists() and c['selection'].get('resume', True):
        state = json.loads(checkpoint.read_text(encoding='utf-8'))
        if state.get('config_hash') != fingerprint(c) or state.get('data_hash') != _data_hash(groups):
            raise ValueError('Resume config/data mismatch; change experiment.run_name or remove old checkpoint.')
        logging.info('Resuming class=%s: %d bootstrap, %d accepted, %d tried', cls,
                     len(state['bootstrap']), len(state['accepted']), len(state['attempted']))
    else:
        state = _new_state(c, groups)
    # Completed runs resume without expensive DINO feature extraction / coverage rebuild.
    if state.get('stop_reason') is not None and state.get('final_pca_dim') is not None:
        logging.info('%s: completed checkpoint, re-exporting results without recomputation', cls)
        _export_class(c, cls, groups, state, out)
        return state

    lookup = {r.id: r for arr in groups.values() for r in arr}
    candidates = {r.id: r for r in groups['candidate']}
    initial = [r.id for r in groups['initial']]
    eligible = set(candidates) - set(state['bootstrap']) - set(state['attempted'])
    chosen = list(state['bootstrap']) + list(state['accepted'])
    budget = int(c['selection']['max_new_images_per_class'])
    mode = c['selection']['mode']
    needs_coverage = mode in ('two_stage', 'coverage_only') or c['bootstrap']['method'] == 'coverage'
    # Fail BEFORE extracting many potentially expensive DINO features.
    b_min = int(c['bootstrap']['min_images']) if c['bootstrap']['enabled'] == 'auto' else 0
    needed_before = max(0, b_min - len(initial) - len(state['bootstrap']))
    if needed_before > budget - len(chosen):
        raise ValueError(f'{cls}: bootstrap requires {needed_before} additions but only '
                         f'{budget - len(chosen)} budget slots remain')
    if needed_before > sum(candidates[i].label in GOOD_LABELS for i in eligible):
        raise ValueError(f'{cls}: bootstrap_insufficient_verified_good; '
                         'label candidates in a manifest before running full feature extraction')
    if c['bootstrap']['enabled'] != 'auto' and c['bootstrap']['enabled'] is not False:
        raise ValueError('bootstrap.enabled must be auto or false')
    if not initial and not chosen and b_min == 0:
        raise ValueError(f'{cls}: no initial PCA images and bootstrap is disabled')
    # Coverage/Residual ranking require every candidate; other modes only need
    # a subset. FeatureCache.get loads missing records lazily.
    if needs_coverage or mode == 'residual_only':
        features.warm(groups['candidate'])
    repo = StatsRepository(features, lookup)
    coverage = None
    if needs_coverage:
        coverage = CoverageEngine(c, groups['candidate'], features, cls, initial)
        coverage.initialize_coverage([lookup[i] for i in initial + chosen])

    # Reentrant bootstrap: checkpoint after each image; checked against TOTAL new budget.
    needed = max(0, b_min - len(initial) - len(state['bootstrap']))
    if needed > budget - len(chosen):
        raise ValueError(f'{cls}: bootstrap requires {needed} extra images, budget remaining {budget-len(chosen)}')
    for slot in range(needed):
        trusted = sorted(i for i in eligible if candidates[i].label in GOOD_LABELS)
        if not trusted:
            raise ValueError(f'{cls}: bootstrap_insufficient_verified_good (remaining={needed-slot})')
        if c['bootstrap']['method'] == 'coverage':
            p = coverage.propose(trusted)
            if p is None:
                raise ValueError('Coverage bootstrap has no eligible candidate')
            chosen_id = p['image_id']
        else:
            chosen_id = random.Random(int(c['experiment']['seed']) + 100000*(len(state['bootstrap'])+1)).choice(trusted)
        state['bootstrap'].append(chosen_id)
        state['decisions'].append({'image_id': chosen_id, 'phase': 'bootstrap', 'status': 'accept',
                                   'reason': c['bootstrap']['method'] + '_bootstrap'})
        eligible.remove(chosen_id)
        chosen.append(chosen_id)
        if coverage:
            coverage.accept(chosen_id)
        atomic_json(checkpoint, state)

    if not initial and not chosen:
        state['stop_reason'] = 'no_initial_or_bootstrap_images'
        atomic_json(checkpoint, state)
        _export_class(c, cls, groups, state, out)
        return state
    stats = repo.combine(initial + chosen)
    model = fit_from_stats(stats, c['pca'])
    logging.info('%s: PCA dimension=%d, train_patches=%d', cls, model.retained, stats.n)
    gate = None
    baseline = None
    if mode in ('two_stage','random_gate'):
        if c['stage2']['gate_metric'] == 'high_error_ratio' and state['ratio_threshold'] is None:
            configured = c['stage2'].get('ratio_threshold')
            state['ratio_threshold'] = float(configured) if configured is not None else calibrate_ratio_threshold(
                model, groups['reference'], features, float(c['stage2']['ratio_quantile']))
        gate = PCAImprovementGate(c, features, repo, groups['reference'], state['ratio_threshold'])
        baseline = reference_metrics(model, groups['reference'], features,
                                     c['stage2']['gate_metric'], state['ratio_threshold'])

    if mode == 'residual_only' and state['residual_initial_scores'] is None:
        if c['stage2']['gate_metric'] == 'high_error_ratio' and state['ratio_threshold'] is None:
            threshold = c['stage2'].get('ratio_threshold')
            state['ratio_threshold'] = float(threshold) if threshold is not None else calibrate_ratio_threshold(
                model, groups['reference'], features, float(c['stage2']['ratio_quantile']))
        state['residual_initial_scores'] = {
            i: image_error_metric(model.errors(features.get(candidates[i])),
                                   c['stage2']['gate_metric'], state['ratio_threshold'])
            for i in sorted(candidates)}
        atomic_json(checkpoint, state)
    while len(chosen) < budget and eligible:
        iteration = len(state['attempted']) + 1
        if mode in ('two_stage','coverage_only'):
            proposal = coverage.propose(eligible)
            if proposal is None:
                state['stop_reason'] = 'gain_threshold_or_no_candidates'
                break
        elif mode == 'residual_only':
            scores = state['residual_initial_scores']
            selected_id = sorted(eligible, key=lambda i: (-scores[i], i))[0]
            proposal = {'image_id': selected_id, 'residual_score': scores[selected_id]}
        else:
            selected_id = random.Random(int(c['experiment']['seed']) + 1000000 + iteration).choice(sorted(eligible))
            proposal = {'image_id': selected_id}
        image_id = proposal['image_id']
        rec = candidates[image_id]
        eligible.remove(image_id)
        state['attempted'].append(image_id)
        if rec.label not in GOOD_LABELS:
            result = {'status': 'pending' if rec.label == 'unknown' else 'reject',
                      'reason': 'label_unverified' if rec.label == 'unknown' else 'confirmed_abnormal'}
        elif gate:
            trial = gate.evaluate(rec, model, stats, baseline)
            result = {'status': trial.status, 'reason': trial.reason,
                      'pca_delta': trial.delta, 'loss_before': trial.before,
                      'loss_after': trial.after, 'p95_before': trial.p95_before,
                      'p95_after': trial.p95_after}
        else:
            result = {'status': 'accept', 'reason': 'direct_accept_without_pca_gate'}
        state['decisions'].append({'iteration': iteration, 'phase':'proposal', 'image_id': image_id,
                                   'label': rec.label, 'source': rec.source, **proposal, **result})
        if result['status'] == 'accept':
            state['accepted'].append(image_id)
            chosen.append(image_id)
            if gate:
                stats = trial.trial_stats
                model = trial.trial_model
                baseline = reference_metrics(model, groups['reference'], features,
                             c['stage2']['gate_metric'], state['ratio_threshold'])
            else:
                stats = stats.plus(repo.get(image_id))
                # No trial refits for direct acceptance: build one final PCA at end.
            if coverage:
                coverage.accept(image_id)
        atomic_json(checkpoint, state)
        logging.info('%s iteration=%d image=%s decision=%s reason=%s new=%d/%d',
                     cls, iteration, image_id, result['status'], result['reason'], len(chosen), budget)

    if state['stop_reason'] is None:
        state['stop_reason'] = 'max_budget' if len(chosen) >= budget else 'no_candidates'
    state['final_pca_dim'] = (model.retained if gate else fit_from_stats(stats, c['pca']).retained)
    state['runtime_seconds'] = time.monotonic() - start_time
    atomic_json(checkpoint, state)
    _export_class(c, cls, groups, state, out)
    return state
