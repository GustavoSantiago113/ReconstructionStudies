import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Tuple
from utils.modules.bottleneckMobile import ConvBNReLU

class Decoder(nn.Module):
    def __init__(self, low_level_in_channels, low_level_out_channels=48, out_channels=256, num_classes=21):
        super().__init__()
        # reduce low-level channels
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_level_in_channels, low_level_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_out_channels),
            nn.ReLU(inplace=True)
        )
        # refine after concatenation
        self.last_conv = nn.Sequential(
            ConvBNReLU(low_level_out_channels + out_channels, out_channels, kernel_size=3),
            ConvBNReLU(out_channels, out_channels, kernel_size=3),
            nn.Conv2d(out_channels, num_classes, kernel_size=1)
        )

    def forward(self, low_level_feat: torch.Tensor, high_level_feat: torch.Tensor, out_size: Tuple[int,int]) -> torch.Tensor:
        """
        low_level_feat: (B, C_low, H_low, W_low) (stride=4)
        high_level_feat: (B, C_high, H_high, W_high) (stride=16)
        out_size: (H_orig, W_orig) for final upsample
        """
        # Upsample high-level by 4 to match low-level spatial size
        high_up = F.interpolate(high_level_feat, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)
        low_proj = self.conv_low(low_level_feat)
        x = torch.cat([high_up, low_proj], dim=1)
        x = self.last_conv(x)
        # final upsample to original image size
        x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        return x