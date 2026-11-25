#!/usr/bin/env python3
"""Build an UNKNOWN-label candidate manifest for human/ground-truth verification."""
import argparse
import csv
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--candidate-root',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
root=Path(a.candidate_root)
if not root.is_dir():
    raise SystemExit(f'Candidate root missing: {root}')
image_ext={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
rows=[]
for cls in sorted(x for x in root.iterdir() if x.is_dir()):
    for image in sorted(cls.rglob('*')):
        if image.is_file() and image.suffix.lower() in image_ext:
            rows.append([image.relative_to(root).as_posix(),'unknown','unverified_candidate'])
out=Path(a.output)
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',encoding='utf-8',newline='') as f:
    w=csv.writer(f)
    w.writerow(['image_path','label_status','source'])
    w.writerows(rows)
print(f'{len(rows)} candidates listed as unknown in {out}. Verify labels before running PCA selection.')
