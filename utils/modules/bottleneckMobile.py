import torch.nn as nn

class ConvBNReLU(nn.Sequential):
    """Standard Conv → BN → ReLU6 block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    """MobileNetV2 Bottleneck block"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = (self.stride == 1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            # 1x1 expand
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1))
        # 3x3 depthwise
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # 1x1 project
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.ReLU6(inplace=True)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)