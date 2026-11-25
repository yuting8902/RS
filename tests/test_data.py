from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from repselect.data.datasets import build_class_records, discover_classes


def sample(path, seed):
    path.parent.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    Image.fromarray(rng.integers(0,255,(8,8,3),dtype=np.uint8)).save(path)


def config(tmp_path):
    return {'data': {'initial_train_root': str(tmp_path/'A'), 'candidate_root':str(tmp_path/'U'),
            'reference_good_root':str(tmp_path/'R'), 'initial_subdir':'train/good',
            'reference_subdir':'val/good','classes':'auto', 'assume_candidates_verified_good':False,
            'label_manifest':None}}


def test_classwise_paths_and_label_policy(tmp_path):
    sample(tmp_path/'A'/'metal'/'train'/'good'/'a.png',1)
    sample(tmp_path/'R'/'metal'/'val'/'good'/'r.png',2)
    sample(tmp_path/'U'/'metal'/'c.png',3)
    c=config(tmp_path)
    assert discover_classes(c)==['metal']
    parts=build_class_records(c,'metal')
    assert len(parts['initial'])==1
    assert len(parts['candidate'])==1
    assert parts['candidate'][0].label=='unknown'
    c['data']['assume_candidates_verified_good']=True
    assert build_class_records(c,'metal')['candidate'][0].label=='verified_good'


def test_duplicate_cross_split_is_rejected(tmp_path):
    sample(tmp_path/'A'/'wood'/'train'/'good'/'a.png',1)
    sample(tmp_path/'R'/'wood'/'val'/'good'/'r.png',2)
    (tmp_path/'U'/'wood').mkdir(parents=True)
    (tmp_path/'U'/'wood'/'duplicate.png').write_bytes((tmp_path/'A'/'wood'/'train'/'good'/'a.png').read_bytes())
    with pytest.raises(ValueError,match='DATA LEAKAGE'):
        build_class_records(config(tmp_path),'wood')
