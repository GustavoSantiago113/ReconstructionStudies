from utils.modules.SegformerEncoder import mix_transformer
from utils.modules.SegformerDecoder import segformer_head
import torch.nn as nn
import torch.nn.functional as F


class segformer(nn.Module):
    def __init__(self,in_channels, num_classes):
        super().__init__()
        self.backbone = mix_transformer(
            in_chans=in_channels,
            embed_dims=(32, 64, 160, 256),  # Reduced from (64, 128, 320, 512)
            num_heads=(1, 2, 4, 8),         # Slightly reduced
            depths=(2, 2, 6, 2),            # Reduced from (3, 4, 18, 3)
            sr_ratios=(8, 4, 2, 1),
            dropout_p=0.0,
            drop_path_p=0.1
        )
        self.decoder_head = segformer_head(
            in_channels=(32, 64, 160, 256), # Match embed_dims
            num_classes=num_classes,
            embed_dim=128                   # Reduced from 256
        )

        
    def forward(self,image):
        image_hw = image.shape[2:]
        x = self.backbone(image) #: Call Encoder
        x = self.decoder_head(x) #: Call Decoder
        x = F.interpolate(x, size=image_hw, mode='bilinear', align_corners=False) # Interpolate to output size
        return x
