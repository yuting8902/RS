"""Standalone PEFT injection for the exact A1 DoRA qkv/proj architecture.
Does not import any code from SubspaceAD_ or AnomalyVFM_ at runtime.
"""
from torch import nn
from .peft_wrapper import PeftType, PeftTarget, create_peft_wrapper


def add_peft(model, r=64, alpha=1.0, peft_type=PeftType.DORA):
    if peft_type != PeftType.DORA:
        raise NotImplementedError('This standalone package supports the A1 DoRA variant only.')
    for name, child in list(model.named_children()):
        if name == 'patch_embed':
            # EXACT parity: upstream _add_attn_peft deliberately skips patch_embed.
            continue
        if name in ('qkv', 'proj') and isinstance(child, nn.Linear):
            target = PeftTarget.QKV if name == 'qkv' else PeftTarget.PROJ
            setattr(model, name, create_peft_wrapper(child, target, peft_type, r, alpha))
        else:
            add_peft(child, r=r, alpha=alpha, peft_type=peft_type)
