import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class DoubleConv(nn.Module):
    """(conv -> BN -> ReLU) * 2"""
    def __init__(self, in_ch, out_ch, mid_ch=None, use_batchnorm=True):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        layers = []
        layers.append(nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=not use_batchnorm))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(mid_ch))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=not use_batchnorm))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then DoubleConv"""
    def __init__(self, in_ch, out_ch, use_batchnorm=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, use_batchnorm=use_batchnorm)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then DoubleConv. Can use bilinear upsample + Conv or ConvTranspose2d."""
    def __init__(self, in_ch, out_ch, bilinear=True, use_batchnorm=True):
        """
        in_ch: number of channels in input (concatenated) — typically 2 * out_ch when using symmetric U-Net.
        out_ch: number of channels after the double conv.
        """
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            # we reduce number of channels with a conv after upsample to reduce params
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, mid_ch=in_ch//2, use_batchnorm=use_batchnorm)
        else:
            # learnable upsample
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, use_batchnorm=use_batchnorm)

    def forward(self, x1, x2):
        """
        x1: decoder feature (to upsample)
        x2: encoder feature (skip connection)
        """
        x1 = self.up(x1)
        # input sizes can differ by 1 pixel due to pooling/odd dims — center-crop or pad as needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # pad x1 to the size of x2
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1,
                 channels: List[int] = None,
                 bilinear: bool = True,
                 use_batchnorm: bool = True):
        """
        channels: list of feature channel sizes for each level of the encoder.
                  e.g. [64, 128, 256, 512, 1024] (classic) or shorter for lightweight.
        bilinear: whether to use bilinear upsample (True) or ConvTranspose2d (False)
        """
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 512, 1024]

        # Validate channels
        assert len(channels) >= 2, "channels must contain at least two elements."

        self.in_conv = DoubleConv(in_channels, channels[0], use_batchnorm=use_batchnorm)

        # encoder (down path)
        self.downs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.downs.append(Down(channels[i], channels[i+1], use_batchnorm=use_batchnorm))

        # decoder (up path)
        self.ups = nn.ModuleList()
        # We'll pair channels in reversed order; when concatenating, input channels to Up = prev_out + skip_ch
        for i in range(len(channels) - 1, 0, -1):
            in_ch = channels[i] + channels[i-1]
            out_ch = channels[i-1]
            self.ups.append(Up(in_ch, out_ch, bilinear=bilinear, use_batchnorm=use_batchnorm))

        self.out_conv = OutConv(channels[0], out_channels)

        # initialize weights
        self._init_weights()

    def forward(self, x):
        enc_feats = []
        x = self.in_conv(x)
        enc_feats.append(x)
        # encoder
        for down in self.downs:
            x = down(x)
            enc_feats.append(x)
        # bottom (last encoder feature is enc_feats[-1])
        # now decoder
        x = enc_feats[-1]
        # iterate ups with corresponding skip features from encoder
        for i, up in enumerate(self.ups):
            skip = enc_feats[-2 - i]  # take features in reverse order (excluding bottom)
            x = up(x, skip)
        logits = self.out_conv(x)
        return logits

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)