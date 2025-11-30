import torch
import torch.nn as nn
from typing import List, Tuple
from torchvision.ops import roi_align

class RoIAlignPool(nn.Module):
    def __init__(self, output_size=(7,7), spatial_scale=1.0, sampling_ratio=2):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, feature_maps: List[torch.Tensor], proposals: torch.Tensor, image_shapes: Tuple[int, int]):
        """
        feature_maps: list [p2..p5] tensors (we will use single-level RoIAlign per proposal per heuristic)
        proposals: Tensor (N,4) in x1,y1,x2,y2 absolute coords
        Returns: pooled features (N, C, output_h, output_w)
        """
        # Build boxes in format (batch_idx, x1,y1,x2,y2) for roi_align
        # Assumes batch 1 for simplicity
        device = proposals.device
        if proposals.numel() == 0:
            return torch.zeros((0, feature_maps[0].shape[1], self.output_size[0], self.output_size[1]),
                               device=device)

        # We'll use a simple FPN->level mapping based on box area (like original paper)
        im_h, im_w = image_shapes
        areas = (proposals[:, 2] - proposals[:, 0]) * (proposals[:, 3] - proposals[:, 1])
        # map to levels: k = floor(4 + log2(sqrt(area)/224))
        scale = torch.sqrt(areas)
        levels = torch.clamp((torch.floor(torch.log2(scale / 224.0) + 4)).to(torch.int64), min=2, max=5)
        # roi_align expects the list of feature maps as a single tensor for each level; we'll accumulate per level
        pooled_per_level = []
        for lvl, feat in enumerate(feature_maps, start=2):  # feature_maps[0] == p2
            inds = torch.nonzero(levels == lvl).squeeze(1)
            if inds.numel() == 0:
                continue
            boxes = proposals[inds]
            batch_indices = boxes.new_zeros((boxes.shape[0], 1))  # assume batch 0
            rois = torch.cat([batch_indices, boxes], dim=1)  # (k,5)
            # compute spatial_scale as ratio of feature map stride to input (level stride)
            # Typical strides: p2=4, p3=8, p4=16, p5=32; compute roughly from feature map shape
            # spatial_scale = feat.shape[-1] / im_w
            stride = int(round(float(im_w) / float(feat.shape[-1])))
            # roi_align uses spatial_scale to map input coords to feature map coords
            spatial_scale = 1.0 / stride
            pooled = roi_align(feat, rois, output_size=self.output_size, spatial_scale=spatial_scale,
                               sampling_ratio=self.sampling_ratio)
            pooled_per_level.append((inds, pooled))
        # Reconstruct ordered pooled features matching proposals order
        pooled_all = proposals.new_zeros((proposals.shape[0], feature_maps[0].shape[1],
                                          self.output_size[0], self.output_size[1])).to(device)
        for inds, pooled in pooled_per_level:
            pooled_all[inds] = pooled
        return pooled_all