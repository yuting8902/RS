#!/usr/bin/env python3
"""Compare two independently exported .npy arrays; imports neither SubspaceAD nor AnomalyVFM."""
import argparse
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--old',required=True,help='Reference features saved outside this project as .npy')
p.add_argument('--new',required=True,help='Selector features saved as .npy')
a=p.parse_args()
x=np.load(a.old,allow_pickle=False)
y=np.load(a.new,allow_pickle=False)
print('old shape:',x.shape,'new shape:',y.shape)
if x.shape!=y.shape:
    raise SystemExit('INCOMPATIBLE SHAPES')
d=np.abs(x.astype('float64')-y.astype('float64'))
print('max_abs_error:',float(d.max()),'mean_abs_error:',float(d.mean()))
print('allclose 1e-5:',bool(np.allclose(x,y,atol=1e-5,rtol=1e-5)))
