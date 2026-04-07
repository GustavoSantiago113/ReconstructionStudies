"""
Interactive per-image SAM2 folder segmentation

Workflow (per image)
  1. Draw a bounding box around the object (click-and-drag).
  2. SAM2 predicts the segmentation mask inside that box.
  3. A polygon editor opens — drag any vertex to refine the boundary,
     then close the window to confirm.
  4. A side-by-side preview shows the original and the white-background result.
     Respond in the terminal:
       [s] save result and move to the next image
       [r] redo this image from the bounding-box step
       [t] tune mask_threshold / iou_threshold for this session
       [q] quit  (run again to resume from where you stopped)

Resume support
  Already-saved images in --output_dir are detected and skipped automatically.

Prerequisites
  pip install git+https://github.com/facebookresearch/sam2.git

How to run (from repository root)
  python utils/segmentation.py \
    --input_dir  data/images/miniature \
    --output_dir data/images/miniature_segmented \
    --sam_checkpoint models/sam2.1_hiera_large.pt

  The SAM2 config is auto-detected from the checkpoint filename.
  Override with --sam_config if needed, e.g.:
    --sam_config configs/sam2.1/sam2.1_hiera_l.yaml

Parameter guidance
  mask_threshold  default 0.0 (= 50% probability logit boundary).
                  Decrease to expand mask coverage; increase to shrink it.
  iou_threshold   default 0.3. Lower values accept lower-confidence candidates.

Outputs
    Segmented PNGs with a transparent background saved to --output_dir.
"""

from pathlib import Path
from typing import Optional, Tuple
import contextlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
import argparse

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor as _SAM2ImagePredictor
    _SAM2_AVAILABLE = True
except Exception:
    _SAM2_AVAILABLE = False

# Maps checkpoint stem substrings to the SAM2 config (relative to sam2/configs/)
_SAM2_CONFIG_MAP = {
    "sam2.1_hiera_large":     "configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_small":     "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_tiny":      "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2_hiera_large":       "configs/sam2/sam2_hiera_l.yaml",
    "sam2_hiera_base_plus":   "configs/sam2/sam2_hiera_b+.yaml",
    "sam2_hiera_small":       "configs/sam2/sam2_hiera_s.yaml",
    "sam2_hiera_tiny":        "configs/sam2/sam2_hiera_t.yaml",
}


def _auto_detect_config(checkpoint_path: str) -> str:
    stem = Path(checkpoint_path).stem
    for key, cfg in _SAM2_CONFIG_MAP.items():
        if key in stem:
            return cfg
    raise ValueError(
        f"Cannot auto-detect SAM2 config from '{stem}'. "
        f"Pass --sam_config explicitly (e.g. configs/sam2.1/sam2.1_hiera_l.yaml)."
    )


class SamFolderSegmenter:
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    @contextlib.contextmanager
    def _infer_ctx(self):
        """No-grad + bfloat16 autocast on CUDA, plain no-grad on CPU."""
        if self.device == "cuda":
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                yield
        else:
            with torch.inference_mode():
                yield

    def _load_sam(self, checkpoint: str, sam_config: Optional[str] = None):
        if not _SAM2_AVAILABLE:
            raise RuntimeError(
                "sam2 is not installed. Install it with:\n"
                "  pip install git+https://github.com/facebookresearch/sam2.git"
            )
        if sam_config is None:
            sam_config = _auto_detect_config(checkpoint)
        print(f"Loading SAM2 — config: {sam_config}")
        sam2 = build_sam2(sam_config, str(checkpoint), device=self.device)
        return _SAM2ImagePredictor(sam2)

    def _set_image(self, predictor, image_np: np.ndarray):
        """Encode image features (cached until next call)."""
        with self._infer_ctx():
            predictor.set_image(image_np)

    def _get_box_from_user(self, pil_img: Image.Image) -> Tuple[int, int, int, int]:
        """Show the image; user clicks and drags to draw a rectangle. Returns (x1, y1, x2, y2).

        Close the window after drawing to confirm the selection.
        """
        from matplotlib.patches import Rectangle as MplRect

        box: list = [None]
        start: list = [None]
        rect_patch: list = [None]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(pil_img)
        ax.set_title("Click and drag to draw a bounding box, then close this window", fontsize=11)
        ax.axis('off')

        def on_press(event):
            if event.inaxes is not ax or event.button != 1:
                return
            start[0] = (event.xdata, event.ydata)
            if rect_patch[0] is not None:
                rect_patch[0].remove()
                rect_patch[0] = None
            fig.canvas.draw_idle()

        def on_motion(event):
            if start[0] is None or event.inaxes is not ax:
                return
            x0, y0 = start[0]
            x1 = event.xdata if event.xdata is not None else x0
            y1 = event.ydata if event.ydata is not None else y0
            if rect_patch[0] is not None:
                rect_patch[0].remove()
            rect_patch[0] = ax.add_patch(MplRect(
                (min(x0, x1), min(y0, y1)), abs(x1 - x0), abs(y1 - y0),
                linewidth=2, edgecolor='lime', facecolor='none'
            ))
            fig.canvas.draw_idle()

        def on_release(event):
            if start[0] is None or event.button != 1:
                return
            x0, y0 = start[0]
            x1 = event.xdata if event.xdata is not None else x0
            y1 = event.ydata if event.ydata is not None else y0
            box[0] = (int(min(x0, x1)), int(min(y0, y1)),
                      int(max(x0, x1)), int(max(y0, y1)))
            ax.set_title(f"Box confirmed: {box[0]}  —  close this window to continue", fontsize=10)
            start[0] = None
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)

        plt.show()

        if box[0] is None:
            raise RuntimeError("No rectangle drawn. Please draw a bounding box around the object.")
        print(f"Box selected: {box[0]}")
        return box[0]

    def _predict_sam_mask(self, predictor, box: Tuple[int, int, int, int],
                          mask_threshold: float = 0.0,
                          iou_threshold: float = 0.3) -> np.ndarray:
        """Predict mask from already-set image features using a bounding box prompt.

        mask_threshold: SAM2 logit threshold (0.0 == 50 % probability);
            lower = larger/more inclusive mask.
        iou_threshold: minimum predicted IoU to accept a mask candidate.
        """
        predictor.mask_threshold = mask_threshold
        with self._infer_ctx():
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array([box[0], box[1], box[2], box[3]]),
                multimask_output=True,
            )
        valid = scores >= iou_threshold
        best_idx = int(np.argmax(scores * valid)) if valid.any() else int(np.argmax(scores))
        return masks[best_idx].astype(np.uint8)

    def _show_overlay(self, pil_img: Image.Image, mask: np.ndarray,
                      box: Optional[Tuple[int, int, int, int]] = None,
                      title: str = "Preview"):
        from matplotlib.patches import Rectangle
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(pil_img)
        overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        overlay[..., 0] = 255
        overlay[..., 3] = (mask * 120).astype(np.uint8)
        ax.imshow(overlay)
        if box is not None:
            x1, y1, x2, y2 = box
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   linewidth=2, edgecolor='lime', facecolor='none'))
        ax.axis('off')
        fig.canvas.manager.set_window_title(title)
        plt.show()

    def _refine_mask(self, mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        if kernel_size <= 1:
            return mask
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        return (mask > 0).astype(np.uint8)

    def _apply_mask_white_bg(self, pil_img: Image.Image, mask: np.ndarray) -> Image.Image:
        # Produce an RGBA image where masked area keeps original colors
        # and background is transparent.
        img = pil_img.convert('RGB')
        arr = np.array(img)
        # If mask and image sizes do not match, resize mask to image size
        if mask.shape != arr.shape[:2]:
            from PIL import Image as PILImage
            mask_img = PILImage.fromarray((mask * 255).astype(np.uint8))
            mask_img = mask_img.resize((arr.shape[1], arr.shape[0]), Image.BILINEAR)
            mask = (np.array(mask_img) / 255.0 > 0.5).astype(np.uint8)
        alpha = (mask * 255).astype(np.uint8)
        rgba = np.dstack([arr, alpha])
        return Image.fromarray(rgba, mode='RGBA')

    # ------------------------------------------------------------------
    # Polygon helpers
    # ------------------------------------------------------------------

    def _mask_to_polygon(self, mask: np.ndarray, max_pts: int = 400) -> Optional[np.ndarray]:
        """Extract the largest contour from mask and resample it to a smooth polygon.

        This returns an Nx2 float array of (x, y) coordinates, or None if no contour.
        The contour is resampled to up to `max_pts` points uniformly along arc-length
        and smoothed with a small moving-average to produce a smoother polygon.
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        contour = contour.reshape(-1, 2).astype(float)
        if contour.shape[0] < 3:
            return None

        # Close contour (ensure last point equals first) for proper arc-length
        if not np.allclose(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]])

        # Compute cumulative arc length
        deltas = np.diff(contour, axis=0)
        seg_lens = np.hypot(deltas[:, 0], deltas[:, 1])
        cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total_len = cumlen[-1]
        if total_len <= 0:
            return contour[:-1]

        # Determine target number of points (at least original, up to max_pts)
        orig_n = contour.shape[0] - 1
        target_n = min(max_pts, max(orig_n, 200))

        # Sample distances uniformly along the contour
        sample_d = np.linspace(0.0, total_len, target_n, endpoint=False)

        # Interpolate x and y along arc-length
        xs = contour[:, 0]
        ys = contour[:, 1]
        xs_interp = np.interp(sample_d, cumlen, xs)
        ys_interp = np.interp(sample_d, cumlen, ys)
        pts = np.stack([xs_interp, ys_interp], axis=1)

        # Smooth via moving average kernel to remove jagged vertices
        if pts.shape[0] >= 5:
            win = min(21, pts.shape[0] // 2 * 2 + 1)  # odd window <=21
            kernel = np.ones(win) / win
            pad = win // 2
            # pad by reflecting to avoid edge shrink
            xs_pad = np.pad(pts[:, 0], pad, mode='reflect')
            ys_pad = np.pad(pts[:, 1], pad, mode='reflect')
            xs_s = np.convolve(xs_pad, kernel, mode='valid')
            ys_s = np.convolve(ys_pad, kernel, mode='valid')
            pts = np.stack([xs_s, ys_s], axis=1)

        return pts.astype(float)

    def _polygon_to_mask(self, polygon: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """Rasterize a polygon (Nx2 float xy) to a binary mask of given (H, W) shape."""
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        return mask

    def _edit_polygon(self, pil_img: Image.Image, mask: np.ndarray) -> np.ndarray:
        """Show the SAM2 mask as a draggable polygon overlay.

        Drag any vertex to refine the mask boundary.
        Close the window to confirm and continue.
        Returns the edited binary mask.
        """
        points = self._mask_to_polygon(mask)
        if points is None or len(points) < 3:
            print("  No editable contour — keeping original mask.")
            return mask

        h, w = np.array(pil_img).shape[:2]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(pil_img)
        ax.set_title(
            "Drag vertices to refine the mask boundary. Close window to confirm.",
            fontsize=10,
        )
        ax.axis('off')

        # Live mask overlay
        ov = np.zeros((h, w, 4), dtype=np.uint8)
        ov[..., 0] = 255
        ov[..., 3] = (mask * 80).astype(np.uint8)
        overlay_img = ax.imshow(ov)

        # Closed polygon line + vertex scatter
        closed = np.vstack([points, points[0]])
        (line,) = ax.plot(closed[:, 0], closed[:, 1], 'g-', linewidth=1.5, zorder=4)
        scat = ax.scatter(
            points[:, 0], points[:, 1],
            c='lime', s=40, zorder=5, edgecolors='white', linewidths=0.5,
        )

        drag_idx: list = [None]

        def on_press(event):
            if event.inaxes is not ax or event.button != 1:
                return
            if event.xdata is None or event.ydata is None:
                return
            dists = np.hypot(points[:, 0] - event.xdata, points[:, 1] - event.ydata)
            idx = int(np.argmin(dists))
            if dists[idx] < 15:
                drag_idx[0] = idx

        def on_motion(event):
            if drag_idx[0] is None or event.inaxes is not ax:
                return
            if event.xdata is None or event.ydata is None:
                return
            points[drag_idx[0]] = [
                float(np.clip(event.xdata, 0, w - 1)),
                float(np.clip(event.ydata, 0, h - 1)),
            ]
            cl = np.vstack([points, points[0]])
            line.set_xdata(cl[:, 0])
            line.set_ydata(cl[:, 1])
            scat.set_offsets(points)
            new_mask = self._polygon_to_mask(points, (h, w))
            new_ov = np.zeros((h, w, 4), dtype=np.uint8)
            new_ov[..., 0] = 255
            new_ov[..., 3] = (new_mask * 80).astype(np.uint8)
            overlay_img.set_data(new_ov)
            fig.canvas.draw_idle()

        def on_release(event):
            drag_idx[0] = None

        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)

        plt.show()
        return self._polygon_to_mask(points, (h, w))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_folder(self,
                       input_dir: str,
                       output_dir: str,
                       sam_checkpoint: str,
                       sam_config: Optional[str] = None,
                       initial_mask_threshold: float = 0.0,
                       initial_iou_threshold: float = 0.3,
                       refine: bool = True,
                       kernel_size: int = 5):
        """Segment images one by one with resume support.

        Per image: draw box → SAM2 mask → polygon editor → preview → save/redo/quit.
        Images that already have a corresponding PNG in output_dir are skipped.
        """
        input_p = Path(input_dir)
        output_p = Path(output_dir)
        output_p.mkdir(parents=True, exist_ok=True)

        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        all_files = sorted([f for f in input_p.iterdir() if f.suffix.lower() in exts])
        if not all_files:
            print("No images found in folder.")
            return

        pending = [f for f in all_files if not (output_p / f"{f.stem}.png").exists()]
        done_count = len(all_files) - len(pending)
        if done_count:
            print(f"Resuming: {done_count}/{len(all_files)} already processed, "
                  f"{len(pending)} remaining.")
        if not pending:
            print("All images already processed.")
            return

        predictor = self._load_sam(sam_checkpoint, sam_config=sam_config)
        mask_threshold = float(initial_mask_threshold)
        iou_threshold = float(initial_iou_threshold)

        for img_idx, f in enumerate(pending):
            print(f"\n[{img_idx + 1}/{len(pending)}] {f.name}  "
                  f"(mask_th={mask_threshold}, iou_th={iou_threshold})")
            pil = Image.open(f).convert('RGB')
            img_np = np.array(pil)
            self._set_image(predictor, img_np)

            while True:
                # Step 1: bounding box
                print("  Draw a bounding box around the object.")
                box = self._get_box_from_user(pil)

                # Step 2: SAM2 prediction
                mask = self._predict_sam_mask(predictor, box,
                                              mask_threshold=mask_threshold,
                                              iou_threshold=iou_threshold)
                if refine:
                    mask = self._refine_mask(mask, kernel_size)

                # Step 3: polygon editing
                print("  Edit polygon vertices if needed, then close the window.")
                mask = self._edit_polygon(pil, mask)

                # Step 4: side-by-side preview
                result = self._apply_mask_white_bg(pil, mask)
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                axes[0].imshow(pil)
                axes[0].set_title("Original")
                axes[0].axis('off')
                axes[1].imshow(result)
                axes[1].set_title("Result (transparent background)")
                axes[1].axis('off')
                fig.suptitle(f"{f.name} — close window, then respond in terminal", fontsize=11)
                plt.show()

                choice = input(
                    "  [s]ave & next, [r]edo, [t]une thresholds, [q]uit: "
                ).strip().lower()

                if choice == 's':
                    out_path = output_p / f"{f.stem}.png"
                    result.save(out_path, 'PNG')
                    print(f"  Saved → {out_path.name}")
                    break
                elif choice == 'r':
                    print("  Redoing segmentation for this image.")
                    continue
                elif choice == 't':
                    try:
                        mt = input(f"    mask_threshold (current {mask_threshold}): ").strip()
                        if mt:
                            mask_threshold = float(mt)
                        it = input(f"    iou_threshold  (current {iou_threshold}): ").strip()
                        if it:
                            iou_threshold = float(it)
                    except ValueError as e:
                        print(f"    Invalid input: {e}")
                    continue
                elif choice == 'q':
                    print("  Stopped. Run again to resume from here.")
                    return
                else:
                    print("  Unknown option.")

        print(f"\nAll done — {len(pending)} images saved to {output_p}")



def parse_args():
    p = argparse.ArgumentParser(
        description="Interactive per-image SAM2 folder segmentation with resume support."
    )
    p.add_argument("--input_dir", required=True, help="Folder of input images.")
    p.add_argument("--output_dir", required=True, help="Folder to save segmented PNGs.")
    p.add_argument("--sam_checkpoint", required=True, help="Path to SAM2 checkpoint.")
    p.add_argument("--sam_config", default=None,
                   help="SAM2 config (auto-detected from checkpoint name if omitted).")
    p.add_argument("--mask_threshold", type=float, default=0.0,
                   help="Logit threshold (default 0.0 = 50%% prob). Lower = larger mask.")
    p.add_argument("--iou_threshold", type=float, default=0.3,
                   help="Min predicted IoU to accept a mask candidate (default 0.3).")
    p.add_argument("--no_refine", action="store_true",
                   help="Disable morphological mask refinement.")
    p.add_argument("--kernel_size", type=int, default=5,
                   help="Morphological kernel size (default 5).")
    return p.parse_args()


def main():
    args = parse_args()
    seg = SamFolderSegmenter()
    seg.process_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sam_checkpoint=args.sam_checkpoint,
        sam_config=args.sam_config,
        initial_mask_threshold=args.mask_threshold,
        initial_iou_threshold=args.iou_threshold,
        refine=not args.no_refine,
        kernel_size=args.kernel_size,
    )


if __name__ == '__main__':
    main()

__all__ = ["SamFolderSegmenter"]
