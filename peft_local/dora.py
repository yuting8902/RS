"""Compatible DoRA implementations for the AnomalyVFM adapted DINOv2 model.
Interface and state_dict parameter names correspond to upstream DoRA wrappers.
"""
import torch
from torch import nn
from torch.nn import functional as F

class _DoRALinearBase(nn.Module):
    def __init__(self, layer, r=64, alpha=1.0):
        super().__init__()
        self.layer = layer
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r if self.r > 0 else 1.0

class DoRAWrapper(_DoRALinearBase):
    def __init__(self, layer, r=64, alpha=1.0):
        super().__init__(layer, r, alpha)
        out_dim, in_dim = layer.weight.shape
        if r > 0:
            self.lora_A = nn.Parameter(torch.randn(r, in_dim) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_dim, r))
            self.magnitude = nn.Parameter(layer.weight.detach().norm(dim=1).clone())
        else:
            self.register_parameter('lora_A', None)
            self.register_parameter('lora_B', None)
            self.register_parameter('magnitude', None)

    @property
    def weight(self):
        return self.layer.weight

    @property
    def bias(self):
        return self.layer.bias

    def forward(self, x):
        if self.r <= 0:
            return self.layer(x)
        adapted = self.layer.weight + (self.lora_B @ self.lora_A) * self.scaling
        direction = adapted / (adapted.norm(dim=1, keepdim=True) + 1e-8)
        return F.linear(x, self.magnitude[:, None] * direction, self.layer.bias)

class DoRAQKVWrapper(_DoRALinearBase):
    def __init__(self, layer, r=64, alpha=1.0):
        super().__init__(layer, r, alpha)
        out_total, in_dim = layer.weight.shape
        if out_total % 3:
            raise ValueError('QKV projection output dimension must be divisible by 3')
        out_dim = out_total // 3
        self.in_features = in_dim
        if r > 0:
            self.lora_A_q = nn.Parameter(torch.randn(r, in_dim) * 0.01)
            self.lora_B_q = nn.Parameter(torch.zeros(out_dim, r))
            self.lora_A_v = nn.Parameter(torch.randn(r, in_dim) * 0.01)
            self.lora_B_v = nn.Parameter(torch.zeros(out_dim, r))
            q, _, v = layer.weight.detach().chunk(3, dim=0)
            self.mag_q = nn.Parameter(q.norm(dim=1).clone())
            self.mag_v = nn.Parameter(v.norm(dim=1).clone())
        else:
            for name in ('lora_A_q','lora_B_q','lora_A_v','lora_B_v','mag_q','mag_v'):
                self.register_parameter(name, None)

    def forward(self, x):
        if self.r <= 0:
            return self.layer(x)
        q, k, v = self.layer.weight.chunk(3, dim=0)
        q_new = q + (self.lora_B_q @ self.lora_A_q) * self.scaling
        v_new = v + (self.lora_B_v @ self.lora_A_v) * self.scaling
        q_out = self.mag_q[:, None] * q_new / (q_new.norm(dim=1, keepdim=True) + 1e-8)
        v_out = self.mag_v[:, None] * v_new / (v_new.norm(dim=1, keepdim=True) + 1e-8)
        return F.linear(x, torch.cat([q_out, k, v_out], dim=0), self.layer.bias)
