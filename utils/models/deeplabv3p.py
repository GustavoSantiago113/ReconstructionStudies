import torch
import torch.nn as nn

from utils.models.mobileNetv2 import MobileNetV2Backbone
from utils.modules.aspp import ASPP
from utils.modules.deepLabDecoder import Decoder

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes: int = 21, output_stride: int = 16, backbone_width_mult: float = 1.0):
        super().__init__()
        assert output_stride in (8, 16), "output_stride must be 8 or 16"
        self.backbone = MobileNetV2Backbone(output_stride=output_stride, width_mult=backbone_width_mult)
        # channels from backbone (for MobileNetV2: low_level ~ 24 (or 16/24 depending on width_mult),
        # high_level last_channel = 1280 * width_mult typically)
        last_channel = int(1280 * backbone_width_mult) if backbone_width_mult > 1.0 else 1280
        # low_level channels — approximate: after the second block group in config above -> 24*width_mult
        # We'll infer dynamically in forward, but need to set a default for constructing decoder.
        # For typical width_mult=1.0, low_level channels is 24. Use 24 * width_mult as default.
        low_level_channels = int(24 * backbone_width_mult)

        # ASPP
        if output_stride == 16:
            atrous_rates = (6, 12, 18)
        else:  # output_stride == 8
            atrous_rates = (12, 24, 36)

        self.aspp = ASPP(last_channel, out_channels=256, atrous_rates=atrous_rates)
        # Decoder
        self.decoder = Decoder(low_level_in_channels=low_level_channels, low_level_out_channels=48,
                               out_channels=256, num_classes=num_classes)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,H,W)
        returns segmentation logits (B, num_classes, H, W)
        """
        orig_size = x.shape[2], x.shape[3]
        feats = self.backbone(x)
        low_level = feats["low_level"]
        high_level = feats["high_level"]

        # ASPP on high-level
        x = self.aspp(high_level)  # (B, 256, H/16, W/16)
        # Decoder fuse and upsample
        x = self.decoder(low_level, x, out_size=orig_size)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)