import torch.nn as nn
from utils.modules.bottleneckMobile import ConvBNReLU, InvertedResidual

class MobileNetV2Backbone(nn.Module):
    """
    MobileNetV2 backbone for DeepLabv3+.
    Outputs:
      - low_level: feature map with stride=4 (good for decoder skip connection)
      - high_level: final feature map (stride=16 or 32)
    """
    def __init__(self, output_stride=16, width_mult=1.0):
        super(MobileNetV2Backbone, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        # Configuration of inverted residual blocks
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],  # low-level feature (stride=4)
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Adjust strides for desired output stride
        current_stride = 1
        dilations = 1
        features = [ConvBNReLU(3, input_channel, stride=2)]  # stride=2

        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                if current_stride >= output_stride:
                    # replace stride with dilation
                    dilation = dilations
                    stride = 1
                    dilations *= 2
                else:
                    dilation = 1
                    current_stride *= stride

                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        self.features = nn.Sequential(*features)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

        self.conv_last = ConvBNReLU(input_channel, self.last_channel, kernel_size=1)

    def forward(self, x):
        low_level = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            # After 2nd block group → low-level feature
            if isinstance(layer, InvertedResidual) and layer.stride == 2 and low_level is None:
                low_level = x
        high_level = self.conv_last(x)
        return {"low_level": low_level, "high_level": high_level}