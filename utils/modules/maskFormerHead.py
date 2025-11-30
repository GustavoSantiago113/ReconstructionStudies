import torch.nn as nn
from torch import Tensor
import torch
from typing import Dict, Tuple
import torch.nn.functional as F

# ------------------------------
# MaskFormer Head (classification + mask projection)
# ------------------------------
class MaskFormerHead(nn.Module):
    """
    Semantic-only head:
      - class logits per query (B, Q, C)
      - query mask logits (B, Q, H, W)
      - aggregated per-class segmentation logits (B, C, H, W)
    """
    def __init__(self, embed_dim=256, num_queries=100, num_classes=21):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.class_embed = nn.Linear(embed_dim, num_classes)  # no background class

        # embeddings used for mask prediction
        self.mask_embed = nn.Linear(embed_dim, embed_dim)
        self.refine = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        # stability helpers
        self.q_norm = nn.LayerNorm(embed_dim)
        self.p_norm = nn.LayerNorm(embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, decoder_output: Tensor, pixel_memory: Tensor, pixel_shape: Tuple[int,int]) -> Dict[str, Tensor]:
        B, Q, C = decoder_output.shape
        H, W = pixel_shape

        # Per-query class logits
        logits_q = self.class_embed(decoder_output)  # (B, Q, C)

        # Normalize query/pixel features to keep dot-products bounded
        q = self.refine(self.mask_embed(decoder_output))     # (B, Q, C)
        q = self.q_norm(q)
        m = self.p_norm(pixel_memory)                        # (B, N, C)
        q = F.normalize(q, dim=-1)
        m = F.normalize(m, dim=-1)

        # Scaled dot-product → per-query masks
        pred_masks = torch.einsum("bqc,bnc->bqn", q, m) * self.scale  # (B, Q, N)
        pred_masks = pred_masks.view(B, Q, H, W)  # (B, Q, H, W)

        # Stable aggregation:
        # weights sum to 1 across queries for each class (prevents exploding sums when C=1)
        weights = F.softmax(logits_q, dim=1)  # (B, Q, C)
        seg_logits = torch.einsum("bqc,bqhw->bchw", weights, pred_masks)  # (B, C, H, W)

        # Clamp to avoid Inf in half precision
        seg_logits = torch.clamp(seg_logits, -20.0, 20.0)
        return {"seg_logits": seg_logits}