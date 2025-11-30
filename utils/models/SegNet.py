from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.modules.SegNetEncoder import vgg16_encoder
from utils.modules.SegNetDecoder import segnet_decoder_head


class segnet(nn.Module):
    """
    Segnet Class:
    - self.backbone: returns (y5, idxs, sizes)
    - self.decoder_head: takes that tuple, returns logits
    - forward(): upsample logits to input spatial size
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        pretrained: bool = True,
        freeze_bn: bool = False,
    ):
        """
        Initialize the SegNet model.

        Args:
            in_channels (int): Number of channels in input image.
            num_classes (int): Number of classes to predict.
            pretrained (bool, optional): Whether to use pre-trained weights. Defaults to True.
            freeze_bn (bool, optional): Whether to freeze batch normalization layers. Defaults to False.
        """
        super().__init__()
        self.backbone = vgg16_encoder(
            in_chans=in_channels, pretrained=pretrained, freeze_bn=freeze_bn
        )
        self.decoder_head = segnet_decoder_head(num_classes=num_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the segnet model.

        Args:
            image (torch.Tensor): Input image of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output logits of shape (B, num_classes, H, W)
        """
        image_hw = image.shape[2:]
        x = self.backbone(image)  # Call Encoder
        x = self.decoder_head(x)  # Call Decoder
        x = F.interpolate(
            x, size=image_hw, mode="bilinear", align_corners=False
        )  # to input size
        return x
