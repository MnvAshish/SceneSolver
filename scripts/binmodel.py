# File: binmodel.py

import torch
import torch.nn as nn
import clip

class CLIPCrimeClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, freeze_clip: bool = True, device: str = "cpu"):
      
        super().__init__()
        # 1) Load CLIP vision model (we ignore text branch here)
        self.clip, _ = clip.load("ViT-B/32", device=device)
        self.visual = self.clip.visual  # outputs 512-dimensional features

        # 2) Optionally freeze CLIP weights
        if freeze_clip:
            for p in self.visual.parameters():
                p.requires_grad = False

        # 3) Attach a small MLP head for our binary task
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      
        # 1) CLIP visual encoder (on CUDA this yields float16)
        feats = self.visual(x)       # e.g. FP16 on GPU

        # 2) Cast back to float32 for your head
        feats = feats.float()        # ← ensure head sees FP32

        # 3) Classification head (always FP32)
        logits = self.head(feats)
        return logits
