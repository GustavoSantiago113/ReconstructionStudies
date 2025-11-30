import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.modules.anchorGenerator import AnchorGenerator
from utils.modules.fastRCNNHead import FastRCNNHead
from utils.modules.featurePyramidModule import FPN
from utils.modules.maskRCNNHead import MaskRCNNMaskHead
from utils.modules.regionProposalNetwork import RegionProposalNetwork
from utils.modules.roiAlignPool import RoIAlignPool


class MaskRCNN(nn.Module):
    def __init__(self, backbone, num_classes=1, fpn_out_channels=128, segmentation_mode=False):
        """
        backbone: should return dict {"c2","c3","c4","c5"} when called with an image tensor.
        num_classes: number of object classes (including background if using classification that way)
        """
        super().__init__()
        self.backbone = backbone
        # C2..C5 channel sizes expected: [256,512,1024,2048] for ResNet101
        in_channels_list = [256, 512, 1024, 2048]
        self.fpn = FPN(in_channels_list, out_channels=fpn_out_channels)
        self.segmentation_mode = segmentation_mode
        # anchor generator for P2..P6
        sizes = ((32,), (64,), (128,), (256,), (512,))
        strides = (4, 8, 16, 32, 64)
        self.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=(0.5,1.0,2.0), strides=strides)
        self.rpn = RegionProposalNetwork(
            self.anchor_generator,
            in_channels=fpn_out_channels,
            pre_nms_top_n=200,      # was 2000
            post_nms_top_n=100,     # was 1000
            nms_thresh=0.7,
            min_size=0
        )
        # RoIAlign pool
        self.roi_pool = RoIAlignPool(output_size=(7,7))
        # Box head and mask head
        self.box_head = FastRCNNHead(in_channels=fpn_out_channels * self.roi_pool.output_size[0] * self.roi_pool.output_size[1],
                                     representation_size=1024, num_classes=num_classes)
        # For mask head we use a larger pooled size (14x14 typical)
        self.mask_roi_pool = RoIAlignPool(output_size=(14,14))
        self.mask_head = MaskRCNNMaskHead(
            in_channels=fpn_out_channels,
            conv_layers=4,
            conv_dim=128,  # was 256
            num_classes=num_classes
        )

    def forward(self, images: torch.Tensor):
        """
        images: Tensor (B,3,H,W).
        If segmentation_mode: returns [B, 1, H, W] for segmentation training.
        Else: returns detection dict/list.
        """
        B, _, H, W = images.shape
        if not self.segmentation_mode:
            # Only support B=1 for detection mode (as before)
            assert B == 1, "Detection mode only supports batch size 1."
            return self._forward_single(images)
        # Segmentation mode: return [B, 1, H, W]
        device = images.device
        seg_masks = torch.zeros((B, 1, H, W), device=device)
        for b in range(B):
            img = images[b:b+1]
            out = self._forward_single(img)
            # out["masks"]: [1, N, 1, H, W] or [N, 1, H, W]
            masks = out["masks"].squeeze(0) if out["masks"].dim() == 5 else out["masks"]
            if masks.numel() > 0:
                # Union of all instance masks (pixel-wise max)
                seg_mask = masks.max(dim=0)[0]  # [1, H, W]
                seg_masks[b] = seg_mask
            # else leave as zeros
        return seg_masks

    def _forward_single(self, images: torch.Tensor):
        """
        images: Tensor (B,3,H,W). Implementation below currently supports B=1 for simplicity.
        Returns: dict with 'boxes', 'scores', 'labels', 'masks' for detections.
        """
        assert images.shape[0] == 1, "Current simple implementation assumes batch size 1 (extendable)."
        device = images.device
        _, _, H, W = images.shape

        # Backbone
        c_feats = self.backbone(images)  # expects dict c2..c5
        # FPN
        p_feats = self.fpn(c_feats)  # dict p2..p6
        p2, p3, p4, p5, p6 = [p_feats[k] for k in ["p2","p3","p4","p5","p6"]]
        feature_maps = [p2, p3, p4, p5, p6]

        # RPN -> proposals (N,4)
        proposals = self.rpn(feature_maps, image_size=(H, W))
        if proposals.numel() == 0:
            return {"boxes": torch.zeros((0,4), device=device),
                    "scores": torch.zeros((0,), device=device),
                    "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                    "masks": torch.zeros((0,1,H,W), device=device)}

        # RoI Pool for box head
        pooled = self.roi_pool([p2,p3,p4,p5], proposals, image_shapes=(H,W))  # (N,C,7,7)
        # flatten channels properly for linear head: the head expects flattened vector length
        # We'll global-average pool then FC to representation size (we created FC expecting flattened)
        # But our FastRCNNHead expects a flattened vector; we will flatten the pooled (N,C,7,7) -> (N, C*7*7)
        scores, bbox_deltas = self.box_head(pooled)
        # For simplicity, assume we have no classification training here: pick top predicted class scores and apply bbox delta for class 1
        probs = F.softmax(scores, dim=1)
        # choose class with highest probability excluding background if background included; our num_classes passed by user
        labels = torch.argmax(probs, dim=1)
        scores_top, _ = torch.max(probs, dim=1)

        # For bbox decode, we need to apply deltas to proposals. bbox_deltas is (N, num_classes*4)
        # For simplicity, apply class-agnostic regression using the first 4 values
        deltas_for_apply = bbox_deltas[:, :4]
        # apply deltas
        final_boxes = self.rpn.apply_deltas(proposals, deltas_for_apply.detach())
        final_boxes = self.rpn.clip_boxes(final_boxes, (H,W))

        # Mask head
        mask_pooled = self.mask_roi_pool([p2,p3,p4,p5], proposals, image_shapes=(H,W))  # (N, C, 14,14)
        mask_logits = self.mask_head(mask_pooled)  # (N, num_classes, Hm, Wm)
        # choose masks for predicted label per roi:
        N = mask_logits.shape[0]
        masks = []
        for i in range(N):
            lbl = labels[i].item()
            m = mask_logits[i, lbl:lbl+1]  # shape (1, Hm, Wm)
            # upsample to original image box size and threshold later; for now, upsample to image size for convenience
            # A proper pipeline would place the mask into the image by cropping/resizing by box coords
            m_up = F.interpolate(m.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
            masks.append(m_up)
        masks = torch.stack(masks, dim=0)  # (N,1,H,W)

        return {
            "boxes": final_boxes.unsqueeze(0),   # [1, N, 4]
            "scores": scores_top.unsqueeze(0),   # [1, N]
            "labels": labels.unsqueeze(0),       # [1, N]
            "masks": masks.unsqueeze(0)          # [1, N, 1, H, W]
        }