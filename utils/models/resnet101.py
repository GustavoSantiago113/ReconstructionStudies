import torch.nn as nn
from utils.modules.bottleneckResNet import Bottleneck

class ResNetBackbone(nn.Module):
    """
    ResNet backbone that returns stage feature maps c2, c3, c4, c5.
    Default configuration = ResNet-101 (layers=(3,4,23,3)).
    """

    def __init__(self, block=Bottleneck, layers=(3, 4, 23, 3),
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, zero_init_residual=False):
        super(ResNetBackbone, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation must be a 3-element tuple/list")

        self.groups = groups
        self.base_width = width_per_group

        # Stem
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])                         # C2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])      # C3
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])      # C4
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])      # C5

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Option to zero-init last BN in each residual branch (improves training stability)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            # replace stride with dilation in later blocks (option used sometimes for segmentation)
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Expected input shape: (B, 3, 512, 512)
        x = self.conv1(x)   # /2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # /4 overall after stem

        c2 = self.layer1(x)  # typically used as stage C2
        c3 = self.layer2(c2) # C3
        c4 = self.layer3(c3) # C4
        c5 = self.layer4(c4) # C5

        # return a dict so it's easy to plug into FPN or Mask R-CNN style heads
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}


def resnet101_backbone(**kwargs):
    """Factory for a ResNet-101 backbone (Bottleneck, layers=(3,4,23,3))."""
    return ResNetBackbone(block=Bottleneck, layers=(3, 4, 23, 3), **kwargs)