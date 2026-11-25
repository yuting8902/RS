import copy
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from repselect.data.datasets import build_class_records
from repselect.runner.class_runner import run_class
from repselect.utils.config import load_configs


class FakeCache:
    def __init__(self, root):
        self.root = root
        self._arrays = {}
    def get(self, rec):
        if rec.id not in self._arrays:
            # Stable synthetic patches: independent of machine, no DINO weights required.
            seed = int(hashlib.sha256(rec.id.encode()).hexdigest()[:8],16)
            rng = np.random.default_rng(seed)
            self._arrays[rec.id] = rng.standard_normal((12,6)).astype(np.float32)
        return self._arrays[rec.id]
    def warm(self, records):
        for r in records:
            self.get(r)


def image(path, seed):
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.random.default_rng(seed).integers(0,255,(6,6,3),dtype=np.uint8)).save(path)


def test_full_controller_and_resume(tmp_path):
    image(tmp_path/'A'/'c1'/'train'/'good'/'a.png',1)
    image(tmp_path/'A'/'c1'/'train'/'good'/'b.png',2)
    image(tmp_path/'R'/'c1'/'val'/'good'/'ref.png',3)
    for j in range(3):
        image(tmp_path/'U'/'c1'/f'{j}.png',10+j)
    from pathlib import Path
    base=Path(__file__).resolve().parents[1]
    c=load_configs(base/'configs/paths.yaml',base/'configs/selection.yaml')
    c['data'].update({'initial_train_root':str(tmp_path/'A'),
                      'candidate_root':str(tmp_path/'U'),
                      'reference_good_root':str(tmp_path/'R'),
                      'assume_candidates_verified_good':True})
    c['output'].update({'root':str(tmp_path/'out'),'cache_root':str(tmp_path/'cache')})
    c['experiment']['run_name']='integration'
    c['pca'].update({'fixed_dim':2,'eigen_device':'cpu'})
    c['stage1'].update({'compute_device':'cpu','compute_dtype':'float32',
                        'reference_chunk_size':8,'candidate_chunk_size':6,
                        'cache_mode':'precompute'})
    c['bootstrap']['min_images']=2
    c['selection']['max_new_images_per_class']=2
    c['stage2']['delta_min']=-999.0  # TEST ONLY: exercise accept/commit; not a science default.
    fake=FakeCache(tmp_path/'fake')
    groups=build_class_records(c,'c1')
    result=run_class(c,'c1',features=fake,groups=groups)
    assert len(result['accepted'])==2
    assert result['stop_reason']=='max_budget'
    assert result['final_pca_dim']==2
    assert len(list((tmp_path/'out'/'integration'/'selected_only'/'c1'/'train'/'good').glob('*.png')))==2
    again=run_class(c,'c1',features=fake,groups=groups)
    assert again['accepted']==result['accepted']
    assert len(again['decisions'])==len(result['decisions'])


def test_all_gate_rejections_do_not_commit(tmp_path):
    image(tmp_path/'A'/'x'/'train'/'good'/'a.png',1)
    image(tmp_path/'R'/'x'/'val'/'good'/'ref.png',2)
    for j in range(2):image(tmp_path/'U'/'x'/f'{j}.png',10+j)
    root=Path(__file__).resolve().parents[1]
    c=load_configs(root/'configs/paths.yaml',root/'configs/selection.yaml')
    c['data'].update({'initial_train_root':str(tmp_path/'A'), 'candidate_root':str(tmp_path/'U'),
                      'reference_good_root':str(tmp_path/'R'),'assume_candidates_verified_good':True})
    c['output'].update({'root':str(tmp_path/'out'),'cache_root':str(tmp_path/'cache')})
    c['pca'].update({'fixed_dim':2,'eigen_device':'cpu'})
    c['stage1'].update({'compute_device':'cpu','compute_dtype':'float32','reference_chunk_size':8,
                        'candidate_chunk_size':6,'cache_mode':'precompute'})
    c['bootstrap']['min_images']=1
    c['selection']['max_new_images_per_class']=2
    c['stage2']['delta_min']=100.0  # IMPOSSIBLE gain threshold to force rejects
    res=run_class(c,'x',features=FakeCache(tmp_path/'f'),groups=build_class_records(c,'x'))
    assert len(res['accepted'])==0
    assert len(res['attempted'])==2
    assert all(d['status']=='reject' for d in res['decisions'])
