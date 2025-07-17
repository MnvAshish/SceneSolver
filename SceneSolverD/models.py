import torch.nn as nn
import clip
import sys
import os

# --- Project Setup ---
# This ensures that imports from the 'scripts' directory work correctly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import constants from the new constants.py file
from scripts.constants import NUM_CLASSES, NUM_BINARY_CLASSES # Import necessary constants

class CLIPCrimeClassifier(nn.Module):
    def __init__(self, num_classes: int, freeze_clip: bool = True, device: str = "cpu"):
        super().__init__()

        # ─── 1) Load CLIP vision encoder ───
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        # ─── Convert CLIP to full precision ───
        self.clip_model = self.clip_model.float()
        self.visual_encoder = self.clip_model.visual  # now in FP32

        if freeze_clip:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False

        # ─── 2) MLP classification head ───
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feats = self.visual_encoder(x)   # x is Float32
        logits = self.classifier(feats)
        return logits
