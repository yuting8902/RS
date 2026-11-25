"""Raw one-image-per-file .npy cache, lazy model initialization, cross-run reuse."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def model_cache_key(c):
    paths = c['model']
    weights = {}
    for key in ('dino_weight_path', 'anomalyvfm_ckpt'):
        p = Path(paths[key])
        if p.exists():
            st = p.stat()
            weights[key] = (str(p.resolve()), st.st_size, st.st_mtime_ns)
        else:
            weights[key] = str(p)
    weights['dino_repo_path'] = str(paths['dino_repo_path'])
    hubconf = Path(paths['dino_repo_path']) / 'hubconf.py'
    if hubconf.exists():
        st = hubconf.stat()
        weights['dino_repo_hubconf'] = (st.st_size, st.st_mtime_ns)
    # Local offline wrapper and DoRA source changes must invalidate cached features.
    project = Path(__file__).resolve().parents[3]
    for rel in ('offline_models/dinov2_offline.py', 'peft_local/dora.py',
                'peft_local/peft_func.py', 'peft_local/peft_wrapper.py'):
        p = project / rel
        if p.is_file():
            weights[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    weights['feature'] = c['feature']
    weights['augmentation'] = c['augmentation']
    return hashlib.sha256(json.dumps(weights, sort_keys=True).encode()).hexdigest()[:20]

class FeatureCache:
    def __init__(self, c, extractor=None):
        self.config = c
        self.root = Path(c['output']['cache_root']).resolve() / 'features' / model_cache_key(c)
        self.root.mkdir(parents=True, exist_ok=True)
        self.extractor = extractor
        self._loaded = {}

    def filename(self, record):
        return self.root / f'{record.digest}.npy'

    def get(self, record):
        p = self.filename(record)
        if not p.is_file():
            if self.extractor is None:
                from .extractor import OfflineFeatureExtractor
                self.extractor = OfflineFeatureExtractor(self.config)
            a = np.asarray(self.extractor.extract(record.path), dtype=np.float32)
            if a.ndim != 2 or a.shape[1] != 1024 or not np.isfinite(a).all():
                raise ValueError(f'Invalid [num_patches,1024] features for {record.id}: {a.shape}')
            tmp = p.with_suffix('.partial.npy')
            np.save(tmp, a)
            os.replace(tmp, p)
        return np.load(p, mmap_mode='r', allow_pickle=False)

    def warm(self, records):
        for i, rec in enumerate(records, 1):
            self.get(rec)
            if i % 50 == 0:
                print(f'  features: {i}/{len(records)}', flush=True)
