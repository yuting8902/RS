"""Safety tests for long-running server jobs: preflight, completed resume, and splits."""
from pathlib import Path
import hashlib
import numpy as np
import pytest
from PIL import Image
from repselect.data.datasets import build_class_records
from repselect.runner.class_runner import run_class
from repselect.utils.config import load_configs


class SyntheticFeatures:
    def __init__(self, root):
        self.root = root
        self.calls = 0

    def get(self, rec):
        self.calls += 1
        seed = int(hashlib.sha256(rec.id.encode()).hexdigest()[:8], 16)
        return np.random.default_rng(seed).normal(size=(10, 6)).astype(np.float32)

    def warm(self, recs):
        for r in recs:
            self.get(r)


class ForbiddenFeatures:
    def warm(self, records):
        raise AssertionError('Finished resume should not extract features')

    def get(self, record):
        raise AssertionError('Finished resume should not extract features')


def save_image(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.default_rng(seed).integers(0, 256, (8, 8, 3),
                dtype=np.uint8)).save(path)


def config_and_data(tmp_path, *, initial=2, candidate=2, verified=True):
    for i in range(initial):
        save_image(tmp_path/'initial'/'c'/'train'/'good'/f'{i}.png', i+1)
    save_image(tmp_path/'reference'/'c'/'val'/'good'/'0.png', 50)
    for i in range(candidate):
        save_image(tmp_path/'candidate'/'c'/f'{i}.png', i+100)
    root = Path(__file__).resolve().parents[1]
    c = load_configs(root/'configs/paths.yaml', root/'configs/selection.yaml')
    c['data'].update({'initial_train_root':str(tmp_path/'initial'),
                      'candidate_root':str(tmp_path/'candidate'),
                      'reference_good_root':str(tmp_path/'reference'),
                      'assume_candidates_verified_good':verified})
    c['output'].update({'root':str(tmp_path/'output'), 'cache_root':str(tmp_path/'cache')})
    c['experiment']['run_name'] = 'test_case'
    c['pca'].update({'fixed_dim':2, 'eigen_device':'cpu'})
    c['stage1'].update({'compute_device':'cpu', 'compute_dtype':'float32',
                        'reference_chunk_size':8, 'candidate_chunk_size':5,
                        'cache_mode':'precompute'})
    c['bootstrap']['min_images']=2
    c['selection']['max_new_images_per_class']=2
    c['stage2']['delta_min']=-100
    return c, build_class_records(c, 'c')


def test_missing_verified_bootstrap_fails_before_extractor(tmp_path):
    c, groups = config_and_data(tmp_path, initial=0, verified=False)
    with pytest.raises(ValueError, match='bootstrap_insufficient_verified_good'):
        run_class(c, 'c', features=ForbiddenFeatures(), groups=groups)


def test_completed_resume_does_not_rerun_expensive_features(tmp_path):
    c, groups = config_and_data(tmp_path)
    c['selection']['mode'] = 'random_only'
    first = run_class(c, 'c', features=SyntheticFeatures(tmp_path/'fake'), groups=groups)
    assert first['stop_reason'] == 'max_budget'
    second = run_class(c, 'c', features=ForbiddenFeatures(), groups=groups)
    assert second['accepted'] == first['accepted']
    assert second['final_pca_dim'] == first['final_pca_dim']


def test_no_resume_cannot_overwrite_existing_job(tmp_path):
    c, groups = config_and_data(tmp_path)
    c['selection']['mode'] = 'random_only'
    run_class(c, 'c', features=SyntheticFeatures(tmp_path/'fake'), groups=groups)
    c['selection']['resume'] = False
    with pytest.raises(FileExistsError, match='Change experiment.run_name'):
        run_class(c, 'c', features=ForbiddenFeatures(), groups=groups)


def test_disable_bootstrap_with_existing_initial_set(tmp_path):
    c, groups = config_and_data(tmp_path, initial=2)
    c['bootstrap']['enabled'] = False
    c['selection']['mode'] = 'random_only'
    result=run_class(c, 'c', features=SyntheticFeatures(tmp_path/'fake'), groups=groups)
    assert not result['bootstrap']
    assert len(result['accepted'])==2
