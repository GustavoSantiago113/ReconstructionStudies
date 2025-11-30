from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.modules.maskFormerHead import MaskFormerHead
from utils.modules.pixelDecoder import PixelDecoder
from utils.modules.transformerEncoderDecoder import PositionEmbeddingSine, SimpleTransformerDecoder, SimpleTransformerEncoder

# ------------------------------
# MaskFormer full model (semantic segmentation only)
# ------------------------------
class MaskFormer(nn.Module):
    """
    Semantic-only MaskFormer:
      - backbone: returns c2..c5 dict
      - pixel_decoder: produces pixel embeddings and mask features
      - encoder: transformer encoder on pixels
      - decoder: transformer decoder with learnable queries
      - head: aggregates query masks into per-class segmentation
    """
    def __init__(self,
                 backbone,
                 num_classes: int = 20,
                 num_queries: int = 100,
                 embed_dim: int = 256,
                 transformer_layers: int = 6,
                 transformer_heads: int = 8,
                 transformer_ffn_dim: int = 2048,
                 return_binary: bool = True):
        """
        backbone: a module that returns dict {"c2","c3","c4","c5"} when called with images
        num_classes: number of semantic classes
        """
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        # Pixel decoder expects channels for c2..c5
        self.pixel_decoder = PixelDecoder(in_channels_list=(256,512,1024,2048), embed_dim=embed_dim, out_stride=4)
        # positional encoding
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=embed_dim // 2)
        # transformer encoder + decoder
        self.encoder = SimpleTransformerEncoder(embed_dim=embed_dim, nhead=transformer_heads, num_layers=transformer_layers,
                                                dim_feedforward=transformer_ffn_dim)
        self.decoder = SimpleTransformerDecoder(embed_dim=embed_dim, nhead=transformer_heads, num_layers=transformer_layers,
                                                dim_feedforward=transformer_ffn_dim)
        # semantic head (no instance/binary branches)
        self.head = MaskFormerHead(embed_dim=embed_dim, num_queries=num_queries, num_classes=num_classes)
        # init weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, images: Tensor) -> Tensor:
        B, _, H, W = images.shape
        feats = self.backbone(images)  # dict c2..c5
        pixel_embeddings, _ = self.pixel_decoder(feats)

        B, C, Hm, Wm = pixel_embeddings.shape
        pixel_flat = pixel_embeddings.flatten(2).permute(0, 2, 1)  # (B, N, C)

        pos = self.pos_embed(pixel_embeddings)  # (B, N, C)
        pixel_with_pos = pixel_flat + pos

        memory = self.encoder(pixel_with_pos)  # (B, N, C)

        # Transformer decoder with learnable queries
        query_embed = self.head.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # (B, Q, C)
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt + query_embed, memory)  # (B, Q, C)

        out = self.head(hs, memory, (Hm, Wm))  # {"seg_logits": (B, C, Hm, Wm)}

        # Upsample to input resolution and return per-class logits
        seg_logits = F.interpolate(out["seg_logits"], size=(H, W), mode="bilinear", align_corners=False)  # (B, C, H, W)
        return seg_logits