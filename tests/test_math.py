from dataclasses import dataclass
import numpy as np
import pytest
from repselect.pca.model import Stats, image_stats, fit_from_stats
from repselect.pca.residual import image_error_metric
from repselect.selection.coverage import CoverageEngine

@dataclass
class FakeRecord:
    id: str
    digest: str

class FakeCache:
    def __init__(self, arrays, root):
        self.arrays=arrays
        self.root=root
    def get(self, rec):
        return self.arrays[rec.id]


def cfg(tmp_path, mode='precompute'):
    return {'stage1':{'compute_device':'cpu','compute_dtype':'float32','reference_chunk_size':3,
                      'candidate_chunk_size':2,'cache_mode':mode,'disk_budget_gib':1,
                      'stop_epsilon':None, 'score':'mean'},
            'output':{'cache_root':str(tmp_path)},'pca':{'mode':'fixed','fixed_dim':2,
               'explained_variance':0.99,'eigen_device':'cpu','whiten':False,'drop_k':0}}


def test_patch_coverage_math_and_reject_no_commit(tmp_path):
    rng=np.random.default_rng(0)
    r=[FakeRecord(f'i{j}',f'h{j}') for j in range(3)]
    x={rec.id: rng.normal(size=(4,4)).astype(np.float32) for rec in r}
    cache=FakeCache(x, tmp_path)
    eng=CoverageEngine(cfg(tmp_path),r,cache,'toy')
    eng.initialize_coverage([])
    p=eng.propose([i.id for i in r])
    assert p['gain']>0
    # Brute force patch-level similarity; fixed U contains the selected candidate itself.
    pool=np.concatenate([x[i.id] for i in r])
    pool=pool/np.linalg.norm(pool,axis=1,keepdims=True)
    candidate=x[p['image_id']]
    candidate=candidate/np.linalg.norm(candidate,axis=1,keepdims=True)
    expected=np.mean(((np.max(pool@candidate.T,axis=1)+1)/2))
    assert abs(p['gain']-expected)<1e-6
    old=eng.base_coverage.copy()
    # Reject = no accept() call, so baseline must remain unchanged.
    assert np.array_equal(old,eng.base_coverage)
    eng.accept(p['image_id'])
    assert np.any(eng.base_coverage>old)
    assert eng.score(p['image_id'])['gain']<1e-7


def test_cache_stream_and_precompute_are_equivalent(tmp_path):
    rng=np.random.default_rng(123)
    records=[FakeRecord(f'j{i}', f'id{i}') for i in range(3)]
    arrays={r.id:rng.normal(size=(5+i,6)).astype(np.float32) for i,r in enumerate(records)}
    cache=FakeCache(arrays,tmp_path)
    full=CoverageEngine(cfg(tmp_path/'full', 'precompute'),records,cache,'cls')
    stream=CoverageEngine(cfg(tmp_path/'stream','stream'),records,cache,'cls')
    full.initialize_coverage([])
    stream.initialize_coverage([])
    for rec in records:
        assert abs(full.score(rec.id)['gain'] - stream.score(rec.id)['gain'])<1e-6
    full.accept(records[0].id)
    stream.accept(records[0].id)
    for rec in records:
        assert abs(full.score(rec.id)['gain'] - stream.score(rec.id)['gain'])<1e-6


def test_pca_statistics_additive_and_residual():
    rng=np.random.default_rng(8)
    a=rng.normal(size=(30,5)).astype(np.float32)
    b=rng.normal(size=(40,5)).astype(np.float32)
    combined=image_stats(a,'cpu').plus(image_stats(b,'cpu'))
    single=image_stats(np.concatenate([a,b]),'cpu')
    assert combined.n==single.n
    assert np.allclose(combined.s,single.s,rtol=1e-12)
    assert np.allclose(combined.q,single.q,rtol=1e-12)
    c={'mode':'fixed','fixed_dim':2,'explained_variance':0.99,'eigen_device':'cpu','whiten':False,'drop_k':0}
    model=fit_from_stats(combined,c)
    centered=np.concatenate([a,b]).astype(np.float64)-model.mu
    recon=centered@model.components@model.components.T
    expected=np.sum((centered-recon)**2,axis=1)
    assert np.allclose(model.errors(np.concatenate([a,b])),expected,atol=1e-9)
    assert model.retained==2


def test_gate_metrics_and_rank_limit():
    assert image_error_metric([1,2,3],'mean')==2
    assert image_error_metric([1,2,3],'p90')==pytest.approx(2.8)
    assert image_error_metric([1,2,3],'high_error_ratio',2)==pytest.approx(1/3)
    bad=Stats(2,np.array([0.,0.,0.]),np.zeros((3,3)))
    with pytest.raises(ValueError):
        fit_from_stats(bad, {'mode':'fixed','fixed_dim':2,'eigen_device':'cpu'})
