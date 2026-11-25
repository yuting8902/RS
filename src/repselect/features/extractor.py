"""Independent A1 offline extractor; not a wrapper around the running SubspaceAD system."""
import logging
from pathlib import Path
import numpy as np

class OfflineFeatureExtractor:
    def __init__(self, config, device='auto'):
        import torch
        from offline_models.dinov2_offline import OfflineAnomalyDINOv2
        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
        self.config = config
        paths = config['model']
        for k in ('dino_repo_path', 'dino_weight_path', 'anomalyvfm_ckpt'):
            if not Path(paths[k]).exists():
                raise FileNotFoundError(f'Missing required model path {k}: {paths[k]}')
        logging.info('Loading independent offline A1 DINOv2 (device=%s)', self.device)
        model = OfflineAnomalyDINOv2(
            dino_repo_path=paths['dino_repo_path'],
            dino_weight_path=paths['dino_weight_path'],
            image_size=int(config['feature']['image_size']),
            peft_rank=64,
            load_base_weights=True,
        )
        model.add_anomalyvfm_dora()
        model.load_anomalyvfm_checkpoint(paths['anomalyvfm_ckpt'], strict=True)
        self.model = model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.transform = self.model.get_img_transform()
        self.feature_dim = int(self.model.feature_dim)

    def extract(self, path):
        import torch
        from PIL import Image
        with Image.open(path) as pil:
            img = pil.convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            _, patches = self.model(tensor)
        if patches.ndim != 3 or patches.shape[0] != 1 or patches.shape[-1] != 1024:
            raise ValueError(f'Unexpected A1 patch tensor shape: {tuple(patches.shape)}')
        a = patches[0].detach().float().cpu().numpy().astype(np.float32)
        if not np.isfinite(a).all() or a.shape[0] == 0:
            raise ValueError(f'Invalid DINO features for {path}')
        return a
