import torch.nn as nn

class MaskRCNNMaskHead(nn.Module):
    """
    Small FCN for mask prediction per ROI.
    Input: pooled features (N, C, 14, 14) typically; we'll use 7x7 -> upsample
    """
    def __init__(self, in_channels, conv_layers=4, conv_dim=256, num_classes=21):
        super().__init__()
        layers = []
        last_channels = in_channels
        for _ in range(conv_layers):
            layers.append(nn.Conv2d(last_channels, conv_dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            last_channels = conv_dim
        self.conv_blocks = nn.Sequential(*layers)
        self.deconv = nn.ConvTranspose2d(last_channels, last_channels, kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.mask_predictor = nn.Conv2d(last_channels, num_classes, kernel_size=1)
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.relu(self.deconv(x))
        masks = self.mask_predictor(x)  # (N, num_classes, H*2, W*2)
        return masks