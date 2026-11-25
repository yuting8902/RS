"""MVTec-like originals and simple class/image candidates, with verified-label policy.
Class-wise splits: initial/<class>/train/good, candidate/<class>/*,
reference/<class>/val/good. No source data is modified.
"""
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
GOOD_LABELS = {'verified_good', 'dataset_good'}

@dataclass(frozen=True)
class ImageRecord:
    id: str
    class_name: str
    path: str
    split: str
    label: str
    source: str
    digest: str


def images_at(root):
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted((p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTS), key=lambda x: str(x))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def discover_classes(c):
    root = Path(c['data']['candidate_root']).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f'Candidate root missing: {root}')
    requested = c['data'].get('classes', 'auto')
    classes = sorted(p.name for p in root.iterdir() if p.is_dir()) if requested == 'auto' else list(requested)
    if not classes:
        raise ValueError(f'No classes found in {root}')
    return classes


def load_label_manifest(path):
    if not path:
        return {}
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Label manifest missing: {path}')
    labels = {}
    with path.open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            rel = (row.get('image_path') or '').replace('\\', '/').lstrip('./')
            if not rel:
                raise ValueError('Each manifest row requires image_path')
            if rel in labels:
                raise ValueError(f'Duplicate manifest image_path: {rel}')
            label = (row.get('label_status') or 'unknown').strip()
            if label not in GOOD_LABELS | {'unknown', 'confirmed_abnormal'}:
                raise ValueError(f'Invalid label_status={label} for {rel}')
            labels[rel] = {'label': label, 'source': row.get('source', 'unspecified')}
    return labels


def build_class_records(c, cls, manifest=None):
    manifest = manifest if manifest is not None else load_label_manifest(c['data'].get('label_manifest'))
    data = c['data']
    paths = {
        'initial': Path(data['initial_train_root']) / cls / data.get('initial_subdir', 'train/good'),
        'candidate': Path(data['candidate_root']) / cls,
        'reference': Path(data['reference_good_root']) / cls / data.get('reference_subdir', 'val/good'),
    }
    if not paths['reference'].is_dir():
        raise FileNotFoundError(f'R_good required for PCA Gate: {paths["reference"]}')
    if not paths['candidate'].is_dir():
        raise FileNotFoundError(f'Candidate directory missing: {paths["candidate"]}')
    records = {}
    global_digests = {}
    for split, root in paths.items():
        rows = []
        for p in images_at(root):
            rel = p.relative_to(root).as_posix()
            image_id = f'{split}:{cls}/{rel}'
            digest = sha256_file(p)
            if digest in global_digests:
                previous = global_digests[digest]
                if previous.split != split:
                    raise ValueError(f'DATA LEAKAGE: duplicate file across {previous.split} and {split}: {previous.path} / {p}')
                # Same split duplicates: deterministic first file only.
                continue
            if split == 'candidate':
                lookup = f'{cls}/{rel}'
                meta = manifest.get(lookup, {})
                label = meta.get('label', 'verified_good' if data.get('assume_candidates_verified_good', False) else 'unknown')
                source = meta.get('source', 'candidate_folder')
            else:
                label = 'dataset_good'
                source = split
            rec = ImageRecord(image_id, cls, str(p.resolve()), split, label, source, digest)
            rows.append(rec)
            global_digests[digest] = rec
        records[split] = rows
    if not records['candidate']:
        raise ValueError(f'No candidate images for class={cls}')
    if not records['reference']:
        raise ValueError(f'No R_good images for class={cls}')
    return records
