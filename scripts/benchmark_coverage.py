#!/usr/bin/env python3
"""Small synthetic GPU GEMM test before running large real class. No model weights."""
import argparse
import time
import torch
p=argparse.ArgumentParser()
p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
p.add_argument('--ref-chunk',type=int,default=2048)
p.add_argument('--candidate-patches',type=int,default=2304)
p.add_argument('--feature-dim',type=int,default=1024)
p.add_argument('--iterations',type=int,default=10)
a=p.parse_args()
if a.device=='cuda' and not torch.cuda.is_available():
    raise SystemExit('CUDA not available. Run --device cpu for a tiny diagnostic.')
dtype=torch.float16 if a.device=='cuda' else torch.float32
x=torch.randn(a.ref_chunk,a.feature_dim,device=a.device,dtype=torch.float32)
y=torch.randn(a.candidate_patches,a.feature_dim,device=a.device,dtype=torch.float32)
x=torch.nn.functional.normalize(x,dim=1).to(dtype)
y=torch.nn.functional.normalize(y,dim=1).to(dtype)
if a.device=='cuda':
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
t0=time.perf_counter()
with torch.no_grad():
    for _ in range(a.iterations):
        z=(x@y.T).max(dim=1).values
if a.device=='cuda': torch.cuda.synchronize()
t=time.perf_counter()-t0
print('device:',a.device,'dtype:',dtype,'reference_chunk:',a.ref_chunk,'candidate_patches:',a.candidate_patches)
print(f'{t/a.iterations:.4f} s per chunk; {a.ref_chunk*a.candidate_patches*a.iterations/t/1e9:.3f} billion pair-similarities/s')
if a.device=='cuda':
    print(f'peak allocated VRAM: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
