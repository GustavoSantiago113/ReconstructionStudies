import torch
import torch.nn as nn
from typing import List, Dict
import torch.nn.functional as F

class FPN(nn.Module):
    """
    Build a Feature Pyramid Network from C2-C5.
    Produces P2,P3,P4,P5 and an extra P6 (stride 32 -> P6 via 3x3 conv stride 2)
    """
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        """
        in_channels_list: channels for C2, C3, C4, C5 (e.g., [256,512,1024,2048])
        out_channels: channels for each FPN output (commonly 256)
        """
        super().__init__()
        self.inner_convs = nn.ModuleList()
        self.layer_convs = nn.ModuleList()
        for in_ch in in_channels_list:
            self.inner_convs.append(nn.Conv2d(in_ch, out_channels, kernel_size=1))
            self.layer_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        # P6 from C5 by stride-2 3x3 conv (like torchvision)
        self.p6 = nn.Conv2d(in_channels_list[-1], out_channels, kernel_size=3, stride=2, padding=1)

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # features expected keys: c2, c3, c4, c5
        names = ["c2", "c3", "c4", "c5"]
        xs = [features[n] for n in names]
        # build top-down
        last_inner = self.inner_convs[-1](xs[-1])
        results = []
        results.append(self.layer_convs[-1](last_inner))  # p5

        # iterate reversed for c4..c2
        for idx in range(len(xs) - 2, -1, -1):
            inner_lateral = self.inner_convs[idx](xs[idx])
            # upsample last_inner
            inner_top_down = F.interpolate(last_inner, size=inner_lateral.shape[-2:], mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_convs[idx](last_inner))

        # results currently [p2,p3,p4,p5]
        p2, p3, p4, p5 = results
        p6 = self.p6(xs[-1])  # from c5
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": p6}