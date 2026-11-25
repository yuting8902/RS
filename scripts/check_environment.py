#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import argparse
import importlib
from repselect.utils.config import load_configs

p = argparse.ArgumentParser()
p.add_argument('--paths', default=str(ROOT/'configs/paths.yaml'))
p.add_argument('--config', default=str(ROOT/'configs/selection.yaml'))
a = p.parse_args()
c = load_configs(a.paths,a.config)
for name in ('torch','torchvision','numpy','PIL','yaml','scipy'):
    try:
        obj = importlib.import_module(name)
        print(f'{name:16} {getattr(obj,"__version__", "installed")}')
    except Exception as e:
        print(f'{name:16} NOT AVAILABLE: {e}')
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    free,total = torch.cuda.mem_get_info()
    print(f'GPU free/total GiB: {free/2**30:.1f}/{total/2**30:.1f}')
for k,v in c['model'].items():
    print(f'{k}: {v} EXIST={Path(v).exists()}')
for k in ('initial_train_root','candidate_root','reference_good_root'):
    pth = Path(c['data'][k])
    print(f'{k}: {pth} EXIST={pth.exists()}')
