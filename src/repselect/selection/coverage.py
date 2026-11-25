"""MaxHerding-inspired grouped Patch Coverage, original full-patch soft coverage.

m_I(p)=max_{q in patches(I)} (1+cos(p,q))/2 for each p in all U;
G(I|T)=mean_{images x in U} mean_{p in x} max(0,m_I(p)-m_T(p)).
Only accepts (not rejects) update m_T. Exact refers to ALL patches, not bitwise
FP64; float16 GEMM is an explicitly configurable numerical approximation.
"""
import hashlib
from collections import OrderedDict
import json
import logging
import os
import shutil
from pathlib import Path
import numpy as np


def normalized_features(raw):
    x = np.asarray(raw, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(n <= 1e-12):
        raise ValueError('Zero-norm DINO patch encountered; cannot evaluate Cosine')
    return x / n

class CoverageEngine:
    def __init__(self, c, candidates, feature_cache, class_name, base_image_ids=None):
        self.c = c
        self.config = c['stage1']
        self.candidates = list(candidates)
        self.features = feature_cache
        self.ids = [rec.id for rec in candidates]
        self.by_id = {rec.id: rec for rec in candidates}
        self.index = {image_id: i for i, image_id in enumerate(self.ids)}
        self.class_name = class_name
        self.base_ids = list(base_image_ids or [])
        self.n_patches = [int(feature_cache.get(r).shape[0]) for r in candidates]
        self.feature_dim = int(feature_cache.get(candidates[0]).shape[1])
        self.offsets = np.cumsum([0] + self.n_patches).tolist()
        self.total_patches = int(self.offsets[-1])
        if self.total_patches == 0:
            raise ValueError('No patches in coverage pool')
        self.device = self.config.get('compute_device', 'auto')
        try:
            import torch
            self.device = 'cuda' if self.device == 'auto' and torch.cuda.is_available() else ('cpu' if self.device == 'auto' else self.device)
            if self.device == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('Requested CUDA but torch.cuda.is_available() is false')
        except ImportError:
            if self.device != 'cpu':
                raise
        self.compute_dtype = self.config.get('compute_dtype', 'float16')
        if self.device == 'cpu':
            self.compute_dtype = 'float32'
        self._gpu_reference = None
        self._candidate_vectors = OrderedDict()
        self.base_coverage = np.zeros(self.total_patches, dtype=np.float32)
        self.cache_root = Path(c['output']['cache_root']).resolve() / 'coverage' / class_name
        self.cache_root.mkdir(parents=True, exist_ok=True)
        token = {'ids': [(r.id, r.digest, self.n_patches[i]) for i, r in enumerate(candidates)],
                 'feature_root': str(feature_cache.root),
                 'dtype': self.compute_dtype, 'score': 'mean',
                 'ref_chunk': int(self.config['reference_chunk_size'])}
        self.cache_key = hashlib.sha256(json.dumps(token, sort_keys=True).encode()).hexdigest()[:20]
        self.cache_dir = self.cache_root / self.cache_key
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / 'candidate_max_similarity.npy'
        self.progress_path = self.cache_dir / 'row_done.npy'
        self.matrix = None
        est_gib = len(candidates) * self.total_patches * 4 / (1024**3)
        self.estimated_cache_gib = est_gib
        mode = self.config['cache_mode']
        if mode == 'auto':
            limit = float(self.config.get('disk_budget_gib', 48))
            available = shutil.disk_usage(self.cache_dir).free / (1024**3)
            mode = 'precompute' if est_gib <= limit and est_gib * 1.1 <= available else 'stream'
            if mode == 'stream':
                logging.warning('Coverage cache estimated %.1f GiB; selected STREAM fallback. '
                                'For thousands of images in ONE class this can be extremely slow. '
                                'Set cache_mode=precompute and provide sufficient disk to avoid '
                                'recomputing every proposal.', est_gib)
        # Avoid launching an accidental multi-day 5k-single-class stream run.
        # The user can explicitly opt into stream mode once benchmarked.
        if self.config['cache_mode'] == 'auto' and mode == 'stream' and len(self.ids) >= 750:
            raise RuntimeError(
                f'{class_name}: {len(self.ids)} candidates would require {est_gib:.1f} GiB '
                'of max-similarity cache. AUTO refuses expensive stream fallback. '
                'Increase stage1.disk_budget_gib and use a fast disk with cache_mode=precompute; '
                'or explicitly set cache_mode=stream after benchmarking.')
        self.cache_mode = mode
        if mode == 'precompute':
            avail = shutil.disk_usage(self.cache_dir).free / (1024**3)
            if not self.cache_path.exists() and est_gib * 1.05 > avail:
                raise OSError(f'Insufficient disk for {est_gib:.1f} GiB coverage cache; free {avail:.1f} GiB')
        logging.info('Coverage %s: N=%s, patches=%s, estimated cache=%.2f GiB, mode=%s, device=%s',
                     class_name, len(self.ids), self.total_patches, est_gib, mode, self.device)

    def _norm_candidate(self, rec):
        return normalized_features(self.features.get(rec))

    def _setup_gpu_reference(self):
        if self.device != 'cuda' or self._gpu_reference is not None:
            return
        import torch
        free, _ = torch.cuda.mem_get_info()
        bytes_per = 2 if self.compute_dtype == 'float16' else 4
        requested = self.total_patches * self.feature_dim * bytes_per
        # Conservative 65% of currently free VRAM; remaining memory for GEMMs/model.
        if requested > 0.65 * free:
            logging.info('Reference pool not resident: needs %.1f GiB, free %.1f GiB', requested/2**30, free/2**30)
            return
        dtype = torch.float16 if self.compute_dtype == 'float16' else torch.float32
        ref = torch.empty((self.total_patches, self.feature_dim), dtype=dtype, device='cuda')
        for j, rec in enumerate(self.candidates):
            x = self._norm_candidate(rec)
            ref[self.offsets[j]:self.offsets[j+1]] = torch.from_numpy(x).to('cuda', dtype=dtype)
        self._gpu_reference = ref
        logging.info('Reference patches loaded to GPU: %.2f GiB', requested / 2**30)

    def _iter_reference_chunks(self):
        n = int(self.config['reference_chunk_size'])
        if self._gpu_reference is not None:
            for start in range(0, self.total_patches, n):
                yield start, self._gpu_reference[start:start+n]
            return
        # No giant memory allocation; reads normalized valid patches image by image.
        for j, rec in enumerate(self.candidates):
            ref = self._norm_candidate(rec)
            for start in range(0, len(ref), n):
                yield self.offsets[j] + start, ref[start:start+n]

    def max_similarities(self, record):
        """Return max shifted cosine for EACH U patch against ALL candidate patches."""
        import torch
        x = self._norm_candidate(record)
        out = np.empty(self.total_patches, dtype=np.float32)
        self._setup_gpu_reference()
        candidate_chunk_size = int(self.config.get('candidate_chunk_size', 2304))
        dtype = torch.float16 if self.compute_dtype == 'float16' and self.device == 'cuda' else torch.float32
        with torch.inference_mode():
            if self.device == 'cuda':
                c_chunks = [torch.from_numpy(x[i:i+candidate_chunk_size].copy()).to('cuda', dtype=dtype)
                            for i in range(0, len(x), candidate_chunk_size)]
                for start, ref in self._iter_reference_chunks():
                    if not isinstance(ref, torch.Tensor):
                        ref = torch.from_numpy(np.asarray(ref).copy()).to('cuda', dtype=dtype)
                    best = None
                    for cand in c_chunks:
                        sim = ref @ cand.T
                        sm = sim.max(dim=1).values.float()
                        best = sm if best is None else torch.maximum(best, sm)
                    result = ((best.clamp(-1, 1) + 1) * 0.5).cpu().numpy()
                    out[start:start+len(result)] = result
            else:
                for start, ref in self._iter_reference_chunks():
                    ref = np.asarray(ref, dtype=np.float32)
                    best = np.full(len(ref), -1., dtype=np.float32)
                    for st in range(0, len(x), candidate_chunk_size):
                        score = ref @ x[st:st+candidate_chunk_size].T
                        best = np.maximum(best, score.max(axis=1))
                    out[start:start+len(ref)] = (np.clip(best, -1, 1) + 1) * 0.5
        return out

    def precompute(self):
        if self.cache_mode != 'precompute':
            return
        shape = (len(self.ids), self.total_patches)
        if self.cache_path.is_file() and self.progress_path.is_file():
            self.matrix = np.lib.format.open_memmap(self.cache_path, mode='r+')
            done = np.load(self.progress_path, allow_pickle=False)
            if self.matrix.shape != shape or done.shape != (len(self.ids),):
                raise ValueError('Corrupted coverage cache shape; remove this cache directory')
        else:
            self.matrix = np.lib.format.open_memmap(self.cache_path, mode='w+', dtype='float32', shape=shape)
            done = np.zeros(len(self.ids), dtype=bool)
            np.save(self.progress_path, done)
        for i, rec in enumerate(self.candidates):
            if done[i]:
                continue
            self.matrix[i] = self.max_similarities(rec)
            self.matrix.flush()
            done[i] = True
            np.save(self.progress_path, done)
            if (i+1) % 10 == 0 or len(self.ids) <= 10:
                logging.info('Coverage precompute %s: %d/%d', self.class_name, i+1, len(self.ids))

    def vector(self, image_id):
        if self.matrix is not None:
            return np.asarray(self.matrix[self.index[image_id]], dtype=np.float32)
        if image_id in self._candidate_vectors:
            self._candidate_vectors.move_to_end(image_id)
            return self._candidate_vectors[image_id]
        vec = self.max_similarities(self.by_id[image_id])
        # Stream mode MUST NOT hold an N-candidate x U-patch cache in RAM.
        # Keep at most two large max-sim vectors (e.g. proposal and latest).
        self._candidate_vectors[image_id] = vec
        while len(self._candidate_vectors) > 2:
            self._candidate_vectors.popitem(last=False)
        return vec

    def initialize_coverage(self, records):
        self.precompute()
        self.base_coverage[:] = 0
        for record in records:
            if record.id in self.index:
                v = self.vector(record.id)
            else:
                v = self.max_similarities(record)
            np.maximum(self.base_coverage, v, out=self.base_coverage)

    def accept(self, image_id):
        np.maximum(self.base_coverage, self.vector(image_id), out=self.base_coverage)
        if self.cache_mode == 'stream':
            self._candidate_vectors.clear()

    def score(self, image_id):
        v = self.vector(image_id)
        gain = np.maximum(v - self.base_coverage, 0)
        sum_img_avg = 0.
        for j in range(len(self.ids)):
            sum_img_avg += float(gain[self.offsets[j]:self.offsets[j+1]].mean())
        total = sum_img_avg / len(self.ids)
        if image_id in self.index:
            i = self.index[image_id]
            self_part = float(gain[self.offsets[i]:self.offsets[i+1]].mean()) / len(self.ids)
        else:
            self_part = 0.
        return {'gain': total, 'self_gain': self_part,
                'other_gain': max(total-self_part, 0.),
                'self_gain_ratio': self_part/total if total > 0 else 0.}

    def propose(self, eligible_ids):
        best = None
        best_score = None
        for image_id in sorted(eligible_ids):
            score = self.score(image_id)
            if best_score is None or score['gain'] > best_score['gain'] + 1e-12:
                best, best_score = image_id, score
        if best is None:
            return None
        eps = self.config.get('stop_epsilon')
        if eps is not None and best_score['gain'] <= float(eps):
            return None
        return {'image_id': best, **best_score}
