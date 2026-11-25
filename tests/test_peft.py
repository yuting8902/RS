from torch import nn
from peft_local.peft_func import add_peft
from peft_local.peft_wrapper import PeftType
from peft_local.dora import DoRAQKVWrapper, DoRAWrapper


def test_dora_patch_embedding_is_NOT_wrapped():
    class FakeDino(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Module()
            self.patch_embed.proj = nn.Linear(4,4)
            self.blocks = nn.ModuleList([nn.Module()])
            self.blocks[0].attn = nn.Module()
            self.blocks[0].attn.qkv = nn.Linear(4,12)
            self.blocks[0].attn.proj = nn.Linear(4,4)
    m=FakeDino()
    add_peft(m.blocks,r=2,peft_type=PeftType.DORA)
    assert isinstance(m.blocks[0].attn.qkv, DoRAQKVWrapper)
    assert isinstance(m.blocks[0].attn.proj, DoRAWrapper)
    assert isinstance(m.patch_embed.proj, nn.Linear)
