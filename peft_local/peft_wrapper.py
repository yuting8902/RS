"""Self-contained DoRA wrapper interface for the supported A1 DINOv2 model.
Only DoRA is implemented; the offline A1 checkpoint never uses LoRA/VPT.
Parameter names match the source AnomalyVFM DoRA wrappers for strict loading.
"""
from enum import Enum
from .dora import DoRAQKVWrapper, DoRAWrapper

class PeftType(str, Enum):
    DORA = 'dora'
    LORA = 'lora'
    VPT = 'vpt'

class PeftTarget(str, Enum):
    QKV = 'qkv'
    PROJ = 'proj'


def create_peft_wrapper(layer, target, peft_type=PeftType.DORA, r=64, alpha=1.0):
    if peft_type != PeftType.DORA:
        raise NotImplementedError('Only A1 DoRA is supported by the standalone selector.')
    return DoRAQKVWrapper(layer, r, alpha) if target == PeftTarget.QKV else DoRAWrapper(layer, r, alpha)
