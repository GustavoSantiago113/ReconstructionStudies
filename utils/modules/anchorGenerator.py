import torch
import math
from typing import List, Tuple

class AnchorGenerator:
    """
    Simple anchor generator for FPN levels.
    Produces anchors for each feature map level given sizes and aspect ratios.
    """
    def __init__(self, sizes=((32,), (64,), (128,), (256,), (512,)),
                 aspect_ratios=(0.5, 1.0, 2.0), strides=(4, 8, 16, 32, 64)):
        """
        sizes: tuple per level of anchor base sizes
        aspect_ratios: tuple of aspect ratios
        strides: stride of each feature map level w.r.t. input image (P2..P6 default)
        """
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides
        assert len(sizes) == len(strides)

    def generate_anchors_per_location(self, base_size):
        anchors = []
        for size in base_size:  # allow multiple sizes per level
            area = size * size
            for ar in self.aspect_ratios:
                w = math.sqrt(area / ar)
                h = w * ar
                anchors.append([-w / 2, -h / 2, w / 2, h / 2])  # centered at 0
        return torch.tensor(anchors)  # (A,4)

    def grid_anchors(self, feature_map_size: Tuple[int, int], stride: int, base_anchors: torch.Tensor, device):
        fm_h, fm_w = feature_map_size
        shift_x = (torch.arange(0, fm_w, device=device) + 0.5) * stride
        shift_y = (torch.arange(0, fm_h, device=device) + 0.5) * stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
        shifts = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1),
                              shift_x.reshape(-1), shift_y.reshape(-1)], dim=1)  # (K,4)
        A = base_anchors.shape[0]
        K = shifts.shape[0]
        anchors = base_anchors.reshape(1, A, 4) + shifts.reshape(K, 1, 4)
        anchors = anchors.reshape(K * A, 4)
        return anchors  # (K*A,4)

    def num_anchors_per_location(self):
        return len(self.aspect_ratios) * max(len(s) for s in self.sizes)

    def forward(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        feature_maps: list of tensors [P2, P3, P4, P5, P6]
        returns: list of anchors tensors per level (N_anchors_level, 4)
        """
        device = feature_maps[0].device
        anchors_per_level = []
        for i, fm in enumerate(feature_maps):
            _, _, h, w = fm.shape
            base_anchors = self.generate_anchors_per_location(self.sizes[i]).to(device)
            anchors = self.grid_anchors((h, w), self.strides[i], base_anchors, device)
            anchors_per_level.append(anchors)
        return anchors_per_level