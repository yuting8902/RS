from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T

from peft_local.peft_func import add_peft
from peft_local.peft_wrapper import PeftType


class OfflineAnomalyDINOv2(nn.Module):
    """
    Fully-offline DINOv2-L/14-Reg wrapper for AnomalyVFM.

    Loading order
    -------------
    1. Build Meta DINOv2 architecture from a local facebookresearch/dinov2 repo.
    2. Load the local original DINOv2 ViT-L/14-register .pth.
    3. Optionally inject AnomalyVFM DoRA adapters (rank 64 by default).
    4. Load anomalyvfm_dinov2.pkl with strict=True.

    The AnomalyVFM checkpoint contains the wrapper model_state_dict, whose keys
    are prefixed by "net.".  This class intentionally keeps the same public
    attributes as the author's models/dinov2.py:
        net, feature_dim, patch_size, H
    """

    def __init__(
        self,
        dino_repo_path,
        dino_weight_path,
        image_size=672,
        peft_rank=64,
        load_base_weights=True,
    ):
        super().__init__()

        self.dino_repo_path = Path(dino_repo_path).expanduser().resolve()
        self.dino_weight_path = Path(dino_weight_path).expanduser().resolve()

        if not self.dino_repo_path.is_dir():
            raise FileNotFoundError(
                f"Local DINOv2 repo not found: {self.dino_repo_path}"
            )
        if not (self.dino_repo_path / "hubconf.py").is_file():
            raise FileNotFoundError(
                f"hubconf.py not found in local DINOv2 repo: "
                f"{self.dino_repo_path}"
            )

        self.net = torch.hub.load(
            str(self.dino_repo_path),
            "dinov2_vitl14_reg",
            source="local",
            pretrained=False,
        )

        self.feature_dim = 1024
        self.patch_size = 14
        self.H = int(image_size)
        self.peft_rank = int(peft_rank)
        self._peft_added = False

        if load_base_weights:
            self.load_base_dinov2_weights(
                self.dino_weight_path,
                strict=True,
            )

    @staticmethod
    def _torch_load(path):
        """
        Load local PyTorch checkpoints across PyTorch versions.

        Newer PyTorch releases support weights_only.  Older releases do not,
        so fall back cleanly when the keyword is unavailable.
        """
        try:
            return torch.load(
                str(path),
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            return torch.load(
                str(path),
                map_location="cpu",
            )

    @staticmethod
    def _extract_state_dict(obj):
        """
        Accept either a raw state_dict or a common checkpoint container.

        The official DINOv2 .pth is normally already a raw state_dict, while
        AnomalyVFM .pkl stores model_state_dict / decoder_state_dict /
        predictor_state_dict.
        """
        if not isinstance(obj, dict):
            raise TypeError(
                "Checkpoint must be a dict/state_dict, "
                f"got {type(obj).__name__}."
            )

        for key in ("model_state_dict", "state_dict", "model"):
            value = obj.get(key)
            if isinstance(value, dict):
                return dict(value)

        return dict(obj)

    @staticmethod
    def _strip_module_prefix(state_dict):
        """Strip an optional DataParallel 'module.' prefix."""
        if state_dict and all(
            str(key).startswith("module.") for key in state_dict
        ):
            return {
                str(key)[len("module."):]: value
                for key, value in state_dict.items()
            }
        return state_dict

    def load_base_dinov2_weights(self, checkpoint_path, strict=True):
        """
        Load the local original Meta DINOv2 ViT-L/14-register checkpoint.
        """
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Local DINOv2 base weight not found: {checkpoint_path}"
            )

        checkpoint = self._torch_load(checkpoint_path)
        state_dict = self._extract_state_dict(checkpoint)
        state_dict = self._strip_module_prefix(state_dict)

        incompatible = self.net.load_state_dict(
            state_dict,
            strict=strict,
        )
        return incompatible

    def add_anomalyvfm_dora(self):
        """
        Inject the same DoRA adapters used by AnomalyVFM into DINOv2 blocks.
        """
        if self._peft_added:
            raise RuntimeError(
                "AnomalyVFM PEFT adapters have already been added."
            )

        add_peft(
            self.net.blocks,
            r=self.peft_rank,
            peft_type=PeftType.DORA,
        )
        self._peft_added = True

    @staticmethod
    def _remove_verified_legacy_proj_aliases(state_dict):
        """
        Remove legacy duplicate projection aliases only when they are proven to
        be exact duplicates of the canonical DoRA-wrapper keys.

        Older anomalyvfm_dinov2 checkpoints can contain both:
            net.blocks.X.attn.proj.weight
            net.blocks.X.attn.proj.layer.weight

        and likewise for bias.

        The current DoRAWrapper registers the underlying Linear module as
        `proj.layer`, so strict loading sees the old `proj.weight/bias` aliases
        as unexpected keys.  We do NOT use strict=False.  Instead, an alias is
        deleted only if:
          - the canonical .layer key exists,
          - shapes are identical, and
          - torch.equal() confirms identical values.

        Any mismatch raises immediately.
        """
        state_dict = dict(state_dict)

        alias_suffixes = (
            ".attn.proj.weight",
            ".attn.proj.bias",
        )

        for old_key in list(state_dict.keys()):
            if not old_key.endswith(alias_suffixes):
                continue

            if old_key.endswith(".attn.proj.weight"):
                new_key = old_key.replace(
                    ".attn.proj.weight",
                    ".attn.proj.layer.weight",
                )
            else:
                new_key = old_key.replace(
                    ".attn.proj.bias",
                    ".attn.proj.layer.bias",
                )

            if new_key not in state_dict:
                raise KeyError(
                    "Legacy AnomalyVFM checkpoint contains "
                    f"{old_key!r}, but the expected canonical key "
                    f"{new_key!r} is missing. Refusing to drop the key."
                )

            old_tensor = state_dict[old_key]
            new_tensor = state_dict[new_key]

            if not torch.is_tensor(old_tensor) or not torch.is_tensor(new_tensor):
                raise TypeError(
                    f"Expected tensors for duplicate alias check: "
                    f"{old_key!r}, {new_key!r}."
                )

            if old_tensor.shape != new_tensor.shape:
                raise ValueError(
                    f"Legacy alias shape mismatch: {old_key} "
                    f"{tuple(old_tensor.shape)} != {new_key} "
                    f"{tuple(new_tensor.shape)}."
                )

            if not torch.equal(old_tensor, new_tensor):
                raise ValueError(
                    f"Legacy alias values differ between {old_key!r} "
                    f"and {new_key!r}. Refusing to remove either key."
                )

            del state_dict[old_key]

        return state_dict

    def load_anomalyvfm_checkpoint(
        self,
        checkpoint_path,
        strict=True,
    ):
        """
        Load the AnomalyVFM DINOv2 model_state_dict.

        DoRA must be injected before this method is called.
        """
        if not self._peft_added:
            raise RuntimeError(
                "Call add_anomalyvfm_dora() before loading the "
                "AnomalyVFM adapted checkpoint."
            )

        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"AnomalyVFM checkpoint not found: {checkpoint_path}"
            )

        checkpoint = self._torch_load(checkpoint_path)
        if not isinstance(checkpoint, dict):
            raise TypeError(
                "AnomalyVFM checkpoint must be a dictionary."
            )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "AnomalyVFM checkpoint does not contain 'model_state_dict'."
            )

        state_dict = dict(checkpoint["model_state_dict"])
        state_dict = self._strip_module_prefix(state_dict)
        state_dict = self._remove_verified_legacy_proj_aliases(
            state_dict
        )

        # Keep strict loading.  The legacy compatibility handling above is
        # intentionally narrow and verified rather than hiding mismatches.
        incompatible = self.load_state_dict(
            state_dict,
            strict=strict,
        )
        return incompatible

    def get_img_transform(self):
        """
        AnomalyVFM DINOv2 preprocessing:
        Resize to square -> ToTensor -> ImageNet normalization.
        """
        return T.Compose(
            [
                T.Resize((self.H, self.H)),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def forward(self, x):
        """
        Return the same features as the author's DINOv2 wrapper:
        normalized CLS token and normalized patch tokens.
        """
        features = self.net.forward_features(x)
        summary = features["x_norm_clstoken"]
        patch_tokens = features["x_norm_patchtokens"]
        return summary, patch_tokens
