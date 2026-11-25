from pathlib import Path
import torch
from torch import nn
from offline_models.dinov2_offline import OfflineAnomalyDINOv2

class TinyDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Linear(4,4)
        self.blocks = nn.ModuleList([nn.Module()])
        self.blocks[0].attn = nn.Module()
        self.blocks[0].attn.qkv = nn.Linear(4,12)
        self.blocks[0].attn.proj = nn.Linear(4,4)
    def forward_features(self,x):
        return {'x_norm_clstoken':torch.zeros(len(x),1024),
                'x_norm_patchtokens':torch.zeros(len(x),4,1024)}


def test_uploaded_offline_wrapper_strict_checkpoint_load(tmp_path, monkeypatch):
    repo=tmp_path/'dinov2'
    repo.mkdir()
    (repo/'hubconf.py').write_text('# fake hub for contract test')
    path=tmp_path/'base.pth'
    torch.save(TinyDino().state_dict(),path)
    monkeypatch.setattr(torch.hub,'load',lambda *args,**kwargs:TinyDino())
    model=OfflineAnomalyDINOv2(str(repo),str(path),image_size=672,peft_rank=2)
    model.add_anomalyvfm_dora()
    ckpt=tmp_path/'adapter.pkl'
    torch.save({'model_state_dict':model.state_dict()},ckpt)
    reloaded=OfflineAnomalyDINOv2(str(repo),str(path),image_size=672,peft_rank=2)
    reloaded.add_anomalyvfm_dora()
    reloaded.load_anomalyvfm_checkpoint(str(ckpt),strict=True)
    _,patches=reloaded(torch.randn(1,3,8,8))
    assert patches.shape==(1,4,1024)
