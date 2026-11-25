#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import argparse
import numpy as np
from repselect.utils.config import load_configs
from repselect.features.extractor import OfflineFeatureExtractor

p = argparse.ArgumentParser()
p.add_argument('--image', required=True, help='Path to one actual normal image')
p.add_argument('--paths', default=str(ROOT/'configs/paths.yaml'))
p.add_argument('--config', default=str(ROOT/'configs/selection.yaml'))
p.add_argument('--save-npy', default=None, help='Optional feature array output for independent parity comparison')
a = p.parse_args()
c=load_configs(a.paths,a.config)
e=OfflineFeatureExtractor(c)
f=e.extract(a.image)
print('shape:',f.shape,'dtype:',f.dtype,'mean:',float(f.mean()),'std:',float(f.std()),'finite:',bool(np.isfinite(f).all()))
assert f.shape == (2304,1024), 'Expected exact 672/14=48 patch grid for A1'
if a.save_npy:
    from pathlib import Path
    Path(a.save_npy).parent.mkdir(parents=True,exist_ok=True)
    np.save(a.save_npy,f)
    print('saved feature array:',a.save_npy)
