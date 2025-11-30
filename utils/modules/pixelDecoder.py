import torch.nn as nn
from typing import Dict, Tuple, List, Optional
from torch import Tensor
import torch
import torch.nn.functional as F

# ------------------------------
# Utilities
# ------------------------------
def _make_conv(in_ch, out_ch, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

# ------------------------------
# Pixel Decoder (FPN-like -> pixel embeddings)
# ------------------------------
class PixelDecoder(nn.Module):
    """
    Simple Pixel Decoder:
      - Take c2, c3, c4, c5
      - Project each to `embed_dim` with 1x1 conv
      - Upsample and sum top-down (like FPN)
      - Final conv to produce pixel embeddings (H/4 x W/4 or configurable)
    Returns:
      pixel_embeddings: (B, embed_dim, H_out, W_out)
      mask_features: same as pixel_embeddings (used to compute masks)
    """
    def __init__(self, in_channels_list=(256,512,1024,2048), embed_dim: int = 256, out_stride=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_stride = out_stride  # final stride relative to input (commonly 4)
        # 1x1 projectors for each stage
        self.proj_convs = nn.ModuleList([nn.Conv2d(c, embed_dim, kernel_size=1) for c in in_channels_list])
        # refinement convs per scale
        self.smooth_convs = nn.ModuleList([_make_conv(embed_dim, embed_dim) for _ in in_channels_list])
        # final fusion
        self.fuse = _make_conv(len(in_channels_list) * embed_dim, embed_dim)
        # final pixel embedding conv
        self.out_conv = _make_conv(embed_dim, embed_dim, kernel_size=3, padding=1)

    def forward(self, features: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        features: dict with keys c2,c3,c4,c5
        returns:
          pixel_embeddings: (B, embed_dim, H_out, W_out)
          mask_features: same as pixel_embeddings
        """
        xs = [features[k] for k in ["c2","c3","c4","c5"]]
        # project
        proj = [p_conv(x) for p_conv, x in zip(self.proj_convs, xs)]
        # top-down upsample & sum like FPN: start from last
        last = proj[-1]
        accum = [None] * len(proj)
        accum[-1] = self.smooth_convs[-1](last)
        for i in range(len(proj)-2, -1, -1):
            up = F.interpolate(last, size=proj[i].shape[-2:], mode='nearest')
            last = proj[i] + up
            accum[i] = self.smooth_convs[i](last)
        # Resize or crop to a common spatial size (use highest resolution among selected maps, typically c2)
        target_h, target_w = accum[0].shape[-2], accum[0].shape[-1]
        resized = [F.interpolate(a, size=(target_h, target_w), mode='bilinear', align_corners=False) for a in accum]
        concat = torch.cat(resized, dim=1)
        fused = self.fuse(concat)
        pixel_embeddings = self.out_conv(fused)  # (B, embed_dim, H_out, W_out)
        mask_features = pixel_embeddings  # name used in MaskFormer paper
        return pixel_embeddings, mask_features
