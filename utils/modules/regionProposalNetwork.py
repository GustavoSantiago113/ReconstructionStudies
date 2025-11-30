import torch
import torch.nn as nn
from utils.modules.anchorGenerator import AnchorGenerator
from typing import List, Tuple
from torchvision.ops import nms

class RPNHead(nn.Module):
    """
    Simple RPN head applied to each FPN level.
    Predicts objectness logits and bbox regression deltas per anchor.
    """
    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(l.weight, std=0.01)
            if l.bias is not None:
                nn.init.constant_(l.bias, 0)

    def forward(self, x):
        t = self.relu(self.conv(x))
        logits = self.cls_logits(t)
        bbox_reg = self.bbox_pred(t)
        return logits, bbox_reg

# ---------- Region Proposal Network (RPN) ----------
class RegionProposalNetwork(nn.Module):
    """
    Simple RPN to generate proposals from anchors and feature maps.
    Uses torchvision.ops.nms for NMS.
    """
    def __init__(self, anchor_generator: AnchorGenerator, in_channels=256,
                 pre_nms_top_n=6000, post_nms_top_n=1000, nms_thresh=0.7, min_size=0):
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = RPNHead(in_channels, num_anchors=anchor_generator.num_anchors_per_location())
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

    def apply_deltas(self, anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """
        anchors: (N,4) x1,y1,x2,y2
        deltas: (N,4) dx,dy,dw,dh (assumes standard parameterization)
        """
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights

        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h
        return pred_boxes

    def clip_boxes(self, boxes: torch.Tensor, image_shape: Tuple[int, int]) -> torch.Tensor:
        h, w = image_shape
        boxes[:, 0].clamp_(min=0, max=w)
        boxes[:, 2].clamp_(min=0, max=w)
        boxes[:, 1].clamp_(min=0, max=h)
        boxes[:, 3].clamp_(min=0, max=h)
        return boxes

    def filter_small_boxes(self, boxes: torch.Tensor, min_size: float) -> torch.Tensor:
        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        return keep

    def forward(self, features: List[torch.Tensor], image_size: Tuple[int, int]):
        """
        features: list of feature maps [p2,p3,p4,p5,p6] each (B,C,H,W)
        image_size: (height, width)
        Returns: List of proposals per image (for batch size 1 it's a single Tensor Nx4)
        """
        assert len(features) == len(self.anchor_generator.sizes)
        device = features[0].device
        anchors_per_level = self.anchor_generator.forward(features)  # list per level (N_l,4)
        logits_per_level = []
        bbox_reg_per_level = []
        for feat in features:
            logits, bbox_reg = self.head(feat)
            # logits: (B, A, H, W) where A = anchors per loc
            logits_per_level.append(logits)
            bbox_reg_per_level.append(bbox_reg)

        # For simplicity, assume batch size = 1 for demonstration (extend as needed)
        proposals_all = []
        scores_all = []
        for lvl_idx, anchors in enumerate(anchors_per_level):
            # flatten logits and bbox for level
            logits = logits_per_level[lvl_idx]  # (B, A, H, W)
            bbox_reg = bbox_reg_per_level[lvl_idx]  # (B, A*4, H, W)
            B = logits.shape[0]
            A = logits.shape[1]
            H = logits.shape[2]
            W = logits.shape[3]
            # reshape
            logits = logits.permute(0, 2, 3, 1).reshape(B, -1)  # (B, K*A)
            bbox_reg = bbox_reg.permute(0, 2, 3, 1).reshape(B, -1, 4)  # (B, K*A, 4)

            # for batch=1:
            scores = logits[0].sigmoid()  # objectness probability
            deltas = bbox_reg[0]

            # apply deltas to anchors
            proposals = self.apply_deltas(anchors, deltas)
            # clip
            proposals = self.clip_boxes(proposals, image_size)
            # filter small
            keep = self.filter_small_boxes(proposals, self.min_size)
            proposals = proposals[keep]
            scores = scores[keep]
            proposals_all.append(proposals)
            scores_all.append(scores)

        # concat all levels
        proposals_all = torch.cat(proposals_all, dim=0)
        scores_all = torch.cat(scores_all, dim=0)

        # pre NMS top-k
        num_pre = min(self.pre_nms_top_n, scores_all.numel())
        if num_pre <= 0:
            return torch.zeros((0, 4), device=device)
        scores_sorted, idx = scores_all.sort(descending=True)
        idx = idx[:num_pre]
        proposals_pre = proposals_all[idx]

        # NMS
        keep = nms(proposals_pre, scores_sorted[:num_pre], self.nms_thresh)
        keep = keep[:self.post_nms_top_n]
        proposals = proposals_pre[keep]
        scores = scores_sorted[:num_pre][keep]

        # proposals: (M,4)
        return proposals  # for batch=1