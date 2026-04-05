from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

class HashEncoding(nn.Module):
    """Multi-resolution hash encoding for Instant NGP"""
    
    def __init__(self, 
                 n_levels: int = 16,
                 n_features_per_level: int = 2,
                 log2_hashmap_size: int = 19,
                 base_resolution: int = 16,
                 finest_resolution: int = 512):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        
        # Growth factor between levels
        self.b = np.exp((np.log(finest_resolution) - np.log(base_resolution)) / (n_levels - 1))
        
        # Hash tables for each level
        self.embeddings = nn.ModuleList([
            nn.Embedding(2 ** log2_hashmap_size, n_features_per_level)
            for _ in range(n_levels)
        ])
        
        # Initialize embeddings
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -1e-4, 1e-4)
    
    def forward(self, x):
        """
        Args:
            x: [..., 3] input coordinates in range [-1, 1]
        Returns:
            [..., n_levels * n_features_per_level] encoded features
        """
        # Normalize to [0, 1]
        x = (x + 1) / 2
        
        # Clamp to valid range
        x = torch.clamp(x, 0, 1)
        
        encoded = []
        for level in range(self.n_levels):
            resolution = int(self.base_resolution * (self.b ** level))
            
            # Scale coordinates to grid resolution
            scaled = x * (resolution - 1)
            
            # Get grid cell corners (trilinear interpolation)
            grid_coords = torch.floor(scaled).long()
            weights = scaled - grid_coords.float()
            
            # Hash function (simple but effective)
            def hash_func(coords):
                # Ensure coords are in valid range
                coords = coords % resolution
                # Simple hash: XOR of prime-scaled coordinates
                primes = torch.tensor([1, 2654435761, 805459861], device=coords.device, dtype=torch.long)
                hashed = torch.zeros(coords.shape[0], device=coords.device, dtype=torch.long)
                for i in range(3):
                    hashed ^= coords[:, i] * primes[i]
                return hashed % (2 ** self.log2_hashmap_size)
            
            # Trilinear interpolation over 8 corners
            features = torch.zeros(x.shape[0], self.n_features_per_level, device=x.device)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        corner = grid_coords.clone()
                        corner[:, 0] += i
                        corner[:, 1] += j
                        corner[:, 2] += k
                        
                        # Get hash indices
                        hashed_indices = hash_func(corner)
                        
                        # Get features from hash table
                        corner_features = self.embeddings[level](hashed_indices)
                        
                        # Compute interpolation weight
                        w = (weights[:, 0] if i == 1 else (1 - weights[:, 0])) * \
                            (weights[:, 1] if j == 1 else (1 - weights[:, 1])) * \
                            (weights[:, 2] if k == 1 else (1 - weights[:, 2]))
                        
                        features += corner_features * w.unsqueeze(-1)
            
            encoded.append(features)
        
        return torch.cat(encoded, dim=-1)


class PositionalEncoding(nn.Module):
    """Positional encoding for directions (still used for view directions)"""
    
    def __init__(self, num_freqs: int = 4):
        super().__init__()
        self.num_freqs = num_freqs
        freq_bands = 2.0 ** torch.linspace(0, num_freqs-1, num_freqs)
        self.register_buffer('freq_bands', freq_bands)
    
    def forward(self, x):
        """
        Args:
            x: [..., C] input coordinates
        Returns:
            [..., C * (2*num_freqs + 1)] encoded coordinates
        """
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)


class SimpleNeRF(nn.Module):
    """Instant NGP network with hash encoding"""
    
    def __init__(self, 
                 n_levels: int = 16,
                 n_features_per_level: int = 2,
                 log2_hashmap_size: int = 19,
                 base_resolution: int = 16,
                 finest_resolution: int = 512,
                 dir_enc_freqs: int = 4,
                 hidden_dim: int = 64):
        super().__init__()
        
        # Use hash encoding for positions (Instant NGP)
        self.pos_encoding = HashEncoding(
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution
        )
        
        # Use positional encoding for directions
        self.dir_encoding = PositionalEncoding(dir_enc_freqs)
        
        pos_enc_dim = n_levels * n_features_per_level
        dir_enc_dim = 3 * (2 * dir_enc_freqs + 1)
        
        # Density network (smaller and faster for Instant NGP)
        self.density_net = nn.Sequential(
            nn.Linear(pos_enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1)  # +1 for density
        )
        
        # Color network
        self.color_net = nn.Sequential(
            nn.Linear(hidden_dim + dir_enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )
    
    def forward(self, positions, directions):
        """
        Args:
            positions: [N, 3] 3D positions
            directions: [N, 3] view directions
        Returns:
            colors: [N, 3] RGB colors
            densities: [N, 1] volume densities
        """
        # Encode positions
        pos_enc = self.pos_encoding(positions)
        
        # Get features and density
        features = self.density_net(pos_enc)
        # Use softplus for density (allows gradients to flow, standard in NeRF)
        density = F.softplus(features[..., :1] - 1.0)  # -1 bias for initialization
        feature_vec = features[..., 1:]
        
        # Encode directions
        dir_enc = self.dir_encoding(directions)
        
        # Get color
        color_input = torch.cat([feature_vec, dir_enc], dim=-1)
        color = self.color_net(color_input)
        
        return color, density

def sample_pdf(bins, weights, n_samples, det=False):
    """Hierarchical sampling via inverse-CDF of a piecewise-constant PDF.

    Args:
        bins:      (N, M) bin edges (z-value midpoints between coarse samples).
        weights:   (N, M) un-normalised importance weights for each bin.
        n_samples: Number of new samples to draw per ray.
        det:       If True, use uniform spacing instead of random (for inference).
    Returns:
        samples: (N, n_samples) depth values drawn from the PDF.
    """
    weights = weights + 1e-5  # prevent NaN
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (N, M+1)

    if det:
        u = torch.linspace(0.0, 1.0, n_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)

    inds_g = torch.stack([below, above], dim=-1)  # (N, n_samples, 2)
    
    """ cdf_g = torch.gather(cdf, -1, inds_g.reshape(*cdf.shape[:-1], -1)).reshape(*inds_g.shape)
    bins_g = torch.gather(bins, -1, inds_g.reshape(*bins.shape[:-1], -1)).reshape(*inds_g.shape)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples """

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


class SimpleNeRFTrainer:
    """Trainer for simplified NeRF loaded from a COLMAP binary reconstruction."""

    def __init__(
        self,
        colmap_dir: str,
        image_dir: str,
        output_dir: str,
        scale_factor: float = 1.0,
        white_bg: bool = False,
        device: str = None,
    ):
        self.colmap_dir = colmap_dir
        self.image_dir = image_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scale_factor = scale_factor
        self.white_bg = white_bg

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load data from COLMAP binary files
        self.load_data()

        # Initialize model
        self.model = SimpleNeRF().to(self.device)

        # Scene normalization uses the already-normalised camera centres from ColmapNeRFDataset.
        # After normalisation cam centres sit near radius ≈ 1, so we use that directly.
        cam_centers = self.poses[:, :3, 3]
        self.scene_center = cam_centers.mean(dim=0)
        distances = (cam_centers - self.scene_center).norm(dim=1)
        self.scene_radius = torch.clamp(distances.max() * 2.0, min=1.0)
        print(f"Scene centre: {self.scene_center.cpu().numpy()}, radius: {float(self.scene_radius):.3f}")
        print(f"Camera distance range: {distances.min():.3f} – {distances.max():.3f}")

    def normalize_positions(self, pts: torch.Tensor) -> torch.Tensor:
        """Map world-space positions to [-1, 1] using scene_center/radius."""
        return torch.clamp((pts - self.scene_center) / self.scene_radius, -1.0, 1.0)

    def load_data(self):
        """Load images, poses, intrinsics, and rays from COLMAP binary files."""
        dataset = ColmapNeRFDataset(
            colmap_dir=self.colmap_dir,
            image_dir=self.image_dir,
            scale_factor=self.scale_factor,
            white_bg=self.white_bg,
            device=str(self.device),
        )

        self.images    = dataset.images        # (N, H, W, 3)  float32 in [0, 1]
        self.poses     = dataset.c2w           # (N, 4, 4)  scene-normalised c2w
        self.K_tensors = dataset.K             # (N, 3, 3)  per-image intrinsics
        self.near      = dataset.near
        self.far       = dataset.far
        self.img_h     = dataset.H
        self.img_w     = dataset.W
        self.N         = dataset.N

        # Scalar intrinsics for the first camera (used as fallback in get_rays)
        self.fx = float(dataset.K[0, 0, 0])
        self.fy = float(dataset.K[0, 1, 1])
        self.cx = float(dataset.K[0, 0, 2])
        self.cy = float(dataset.K[0, 1, 2])

        # Precomputed rays – avoids recomputing per training step
        self._rays_o  = dataset.rays_o    # (N·H·W, 3)
        self._rays_d  = dataset.rays_d    # (N·H·W, 3)
        self._targets = dataset.targets   # (N·H·W, 3)

        print(f"Loaded {self.N} images ({self.img_h}×{self.img_w})")
        print(f"Intrinsics (cam 0): fx={self.fx:.1f} fy={self.fy:.1f} cx={self.cx:.1f} cy={self.cy:.1f}")
        print(f"Near/far: {self.near:.3f} / {self.far:.3f}")
    
    def debug_network_output(self, n_samples: int = 1000):
        """Debug method to check if network produces non-zero outputs"""
        print("\\nDebugging network output...")
        # Sample random positions in normalized space
        pos = torch.randn(n_samples, 3, device=self.device) * 0.5  # Within [-1, 1] roughly
        dirs = torch.randn(n_samples, 3, device=self.device)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)  # Normalize directions
        
        with torch.no_grad():
            colors, densities = self.model(pos, dirs)
            sigma = densities.squeeze()  # Already softplus activated
            
        print(f"  Colors range: [{colors.min():.4f}, {colors.max():.4f}]")
        print(f"  Densities (sigma) range: [{sigma.min():.4f}, {sigma.max():.4f}]")
        print(f"  Non-zero densities: {(sigma > 1e-4).sum().item()}/{n_samples}")
    
    def get_rays(self, pose, H, W):
        """Generate camera rays using fx, fy, cx, cy."""
        i, j = torch.meshgrid(
            torch.arange(W, dtype=torch.float32, device=self.device),
            torch.arange(H, dtype=torch.float32, device=self.device),
            indexing='xy'
        )
        
        dirs = torch.stack([
            (i - self.cx) / self.fx,
            -(j - self.cy) / self.fy,
            -torch.ones_like(i)
        ], dim=-1)
        
        # Rotate ray directions from camera frame to world frame
        rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)
        
        # Origin is the camera position
        #rays_o = pose[:3, 3].expand(rays_d.shape)
        rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
        return rays_o, rays_d
    
    def render_rays(self, rays_o, rays_d, near=None, far=None, n_samples=64, n_importance=64):
        """Volume rendering along rays with hierarchical sampling.

        Parameters
        ----------
        n_samples    : Coarse (uniform) samples per ray.
        n_importance : Additional importance samples per ray (0 to disable).
        """
        if near is None:
            near = self.near
        if far is None:
            far = self.far

        N_rays = rays_o.shape[0]

        # --- Coarse uniform sampling ---
        t_vals = torch.linspace(0, 1, n_samples, device=self.device)
        z_vals = near * (1 - t_vals) + far * t_vals
        z_vals = z_vals.expand(N_rays, n_samples)

        # Stratified jittering during training
        if self.model.training:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            z_vals = lower + (upper - lower) * torch.rand_like(z_vals)

        def _volume_render(z):
            """Query the network at z-depths and return (rgb, weights)."""
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z[..., :, None]
            pts_flat = pts.reshape(-1, 3)
            pts_flat_norm = self.normalize_positions(pts_flat)

            dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            dirs_flat = dirs[:, None, :].expand_as(pts).reshape(-1, 3)

            colors, densities = self.model(pts_flat_norm, dirs_flat)
            colors = colors.reshape(*pts.shape[:-1], 3)
            densities = densities.reshape(*pts.shape[:-1])

            dists = z[..., 1:] - z[..., :-1]
            dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e10], dim=-1)

            sigma = densities.squeeze(-1)
            alpha = 1.0 - torch.exp(-sigma * dists)
            weights = alpha * torch.cumprod(
                torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
                dim=-1,
            )[..., :-1]

            rgb = torch.sum(weights[..., None] * colors, dim=-2)
            return rgb, weights

        # Coarse pass
        _, coarse_weights = _volume_render(z_vals)

        # --- Importance sampling (fine pass) ---
        if n_importance > 0:
            z_mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_mids,
                coarse_weights[..., 1:-1].detach(),
                n_importance,
                det=not self.model.training,
            )
            z_vals_fine, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
        else:
            z_vals_fine = z_vals

        # Final rendering with combined samples
        rgb, weights = _volume_render(z_vals_fine)

        return rgb, weights, z_vals_fine

    @staticmethod
    def depth_concentration_loss(weights, z_vals):
        """Encourage concentrated density along rays (improves depth sharpness).

        Computes the weighted variance of depth per ray.  Minimising this
        pushes the weight distribution towards a single surface, eliminating
        floaters and producing cleaner depth maps.
        """
        depth_mean = (weights * z_vals).sum(dim=-1, keepdim=True)  # (N, 1)
        depth_var = (weights * (z_vals - depth_mean) ** 2).sum(dim=-1)  # (N,)
        return depth_var.mean()

    def render_image(self, img_idx: int = 0, n_samples: int = 64, chunk: int = 8192):
        """Render a full (H, W) image for the given image index."""
        pose = self.poses[img_idx]
        H, W = self.img_h, self.img_w
        # Use per-image intrinsics
        K = self.K_tensors[img_idx]
        fx = float(K[0, 0]); fy = float(K[1, 1])
        cx = float(K[0, 2]); cy = float(K[1, 2])
        i_grid, j_grid = torch.meshgrid(
            torch.arange(W, dtype=torch.float32, device=self.device),
            torch.arange(H, dtype=torch.float32, device=self.device),
            indexing='xy'
        )
        dirs = torch.stack([
            (i_grid - cx) / fx,
            -(j_grid - cy) / fy,
            -torch.ones_like(i_grid)
        ], dim=-1)
        rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)
        rays_o = pose[:3, 3].expand(rays_d.shape)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        all_rgb, all_depth = [], []
        with torch.no_grad():
            for i in range(0, rays_o.shape[0], chunk):
                ro = rays_o[i:i+chunk]
                rd = rays_d[i:i+chunk]
                rgb, weights, z_vals = self.render_rays(ro, rd, n_samples=n_samples)
                depth = (weights * z_vals).sum(dim=-1)
                all_rgb.append(rgb)
                all_depth.append(depth)
        rgb   = torch.cat(all_rgb,   dim=0).reshape(H, W, 3).clamp(0, 1).cpu().numpy()
        depth = torch.cat(all_depth, dim=0).reshape(H, W).cpu().numpy()
        return rgb, depth

    def render_preview_figure(
        self,
        img_idx: int = 0,
        n_samples: int = 64,
        step: int = 0,
        loss: float = 0.0,
    ):
        """Render *img_idx* and return a matplotlib Figure comparing GT vs predicted."""
        import matplotlib.pyplot as plt  # local import avoids hard dep at module level
        self.model.eval()
        rgb, depth = self.render_image(img_idx, n_samples=n_samples)
        gt = self.images[img_idx].cpu().numpy().clip(0, 1)
        psnr = float(-10.0 * np.log10(((rgb - gt) ** 2).mean() + 1e-8))

        d_min, d_max = depth.min(), depth.max()
        depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].imshow(gt);           axes[0].set_title("Ground truth")
        axes[1].imshow(rgb.clip(0,1));axes[1].set_title(f"Predicted  PSNR={psnr:.2f} dB")
        axes[2].imshow(depth_norm, cmap="turbo"); axes[2].set_title("Depth")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"Step {step}  |  loss={loss:.5f}")
        fig.tight_layout()
        return fig, psnr

    def save_preview(self, img_idx: int = 0, n_samples: int = 64, prefix: str = "preview"):
        rgb, depth = self.render_image(img_idx, n_samples=n_samples)
        rgb_img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        d_norm = depth / (depth.max() + 1e-8)
        d_img = (d_norm * 255).astype(np.uint8)
        Image.fromarray(rgb_img).save(self.output_dir / f"{prefix}_rgb_{img_idx:03d}.png")
        Image.fromarray(d_img).save(self.output_dir / f"{prefix}_depth_{img_idx:03d}.png")
        print(f"  Saved: {prefix}_rgb_{img_idx:03d}.png")
    
    def train(
        self,
        n_iters: int = 10000,
        batch_size: int = 1024,
        lr: float = 5e-4,
        lr_decay: float = 0.1,
        log_every: int = 100,
        preview_every: int = 0,
        preview_index: int = 0,
        preview_n_samples: int = 64,
        dist_loss_weight: float = 0.01,
    ):
        """Train using pre-computed rays from the COLMAP dataset.

        Parameters
        ----------
        n_iters         : Total training steps.
        batch_size      : Rays per training step.
        lr              : Initial Adam learning rate.
        lr_decay        : Multiplicative final LR = lr * lr_decay (exponential).
        log_every       : Print loss/PSNR every N steps.
        preview_every   : Inline-render a validation image every N steps (0 = off).
        preview_index   : Which dataset image to use for validation previews.
        preview_n_samples: Points sampled per ray during preview rendering.
        dist_loss_weight: Weight for depth-concentration (distortion) loss (0 to disable).
        """
        import matplotlib.pyplot as plt
        from tqdm.auto import trange

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))
        lr_lambda = lambda step: lr_decay ** (step / max(n_iters, 1))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        n_total_rays = self._rays_o.shape[0]
        best_psnr   = -float('inf')
        best_loss   = float('inf')
        best_state  = None
        best_step   = 0
        train_losses, train_psnrs, val_psnrs, val_steps = [], [], [], []

        print(f"\nTraining for {n_iters} steps  |  {n_total_rays:,} total rays  |  lr {lr:.1e}→{lr*lr_decay:.1e}")

        pbar = trange(1, n_iters + 1, desc="Training", dynamic_ncols=True)
        for step in pbar:
            self.model.train()

            # Sample a random batch from ALL pre-computed training rays
            idx = torch.randint(0, n_total_rays, (batch_size,), device=self.device)
            rays_o_b  = self._rays_o[idx]
            rays_d_b  = self._rays_d[idx]
            target_rgb = self._targets[idx]

            rgb, weights, z_vals = self.render_rays(rays_o_b, rays_d_b)

            loss = F.mse_loss(rgb, target_rgb)

            # Depth-concentration loss – sharpens density along rays
            if dist_loss_weight > 0:
                loss = loss + dist_loss_weight * self.depth_concentration_loss(weights, z_vals)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            psnr = float(-10.0 * np.log10(F.mse_loss(rgb.detach(), target_rgb).item() + 1e-8))

            # Update progress bar and logs
            pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.2f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            if step % log_every == 0:
                train_losses.append(loss.item())
                train_psnrs.append(psnr)

            if preview_every > 0 and step % preview_every == 0:
                self.model.eval()
                fig, vpsnr = self.render_preview_figure(
                    preview_index, n_samples=preview_n_samples, step=step, loss=loss.item()
                )
                plt.show()
                val_psnrs.append(vpsnr)
                val_steps.append(step)

                if vpsnr > best_psnr:
                    best_psnr  = vpsnr
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    best_step  = step
                    self.save_best_checkpoint(best_state, best_step, loss.item(), best_psnr)

            # Fallback: track best by loss when no periodic previews
            if preview_every == 0 and loss.item() < best_loss:
                best_loss  = loss.item()
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                best_step  = step

        # Always save at the end if we didn't save via preview
        if best_state is not None and (preview_every == 0 or best_step == 0):
            self.save_best_checkpoint(best_state, best_step, best_loss, best_psnr)

        print(f"\n✓ Training complete!  Best step={best_step}  loss={best_loss:.5f}  PSNR={best_psnr:.2f} dB")
        return {"train_losses": train_losses, "train_psnrs": train_psnrs,
                "val_psnrs": val_psnrs, "val_steps": val_steps}

    def render_from_pose(self, c2w, K=None, n_samples: int = 64, chunk: int = 8192):
        """Render a full image from an arbitrary camera-to-world pose (novel view)."""
        if K is None:
            K = self.K_tensors[0]
        c2w = c2w.to(self.device)
        H, W = self.img_h, self.img_w
        fx = float(K[0, 0]); fy = float(K[1, 1])
        cx = float(K[0, 2]); cy = float(K[1, 2])
        i_grid, j_grid = torch.meshgrid(
            torch.arange(W, dtype=torch.float32, device=self.device),
            torch.arange(H, dtype=torch.float32, device=self.device),
            indexing='xy',
        )
        dirs = torch.stack([
            (i_grid - cx) / fx,
            -(j_grid - cy) / fy,
            -torch.ones_like(i_grid),
        ], dim=-1)
        rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1).reshape(-1, 3)
        rays_o = c2w[:3, 3].expand(rays_d.shape)
        all_rgb, all_depth = [], []
        with torch.no_grad():
            for i in range(0, rays_o.shape[0], chunk):
                rgb, weights, z_vals = self.render_rays(
                    rays_o[i:i+chunk], rays_d[i:i+chunk], n_samples=n_samples
                )
                all_rgb.append(rgb)
                all_depth.append((weights * z_vals).sum(dim=-1))
        rgb   = torch.cat(all_rgb,   dim=0).reshape(H, W, 3).clamp(0, 1).cpu().numpy()
        depth = torch.cat(all_depth, dim=0).reshape(H, W).cpu().numpy()
        return rgb, depth

    def save_best_checkpoint(self, state_dict, step: int, loss: float, psnr: float = 0.0):
        """Save the best model checkpoint."""
        ckpt_path = self.output_dir / "nerf_best.pth"
        torch.save({
            'step': step,
            'loss': loss,
            'psnr': psnr,
            'model_state_dict': state_dict,
        }, ckpt_path)
        print(f"  Saved checkpoint → {ckpt_path}  (step={step}, loss={loss:.5f}, PSNR={psnr:.2f} dB)")


# ---------------------------------------------------------------------------
# COLMAP conversion helpers
# ---------------------------------------------------------------------------

def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert a COLMAP quaternion (qw, qx, qy, qz) to a 3×3 rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=float,
    )


def colmap_to_c2w(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert a COLMAP image pose to a 4×4 camera-to-world matrix.

    COLMAP stores the **world-to-camera** rigid transform:
        p_cam = R_cw @ p_world + t

    The inverse (camera-to-world) is:
        c2w[:3, :3] = R_cw.T
        c2w[:3,  3] = −R_cw.T @ t   (= camera centre in world space)
    """
    R_cw = qvec_to_rotmat(qvec)
    t = np.asarray(tvec, dtype=float)
    c2w = np.eye(4, dtype=float)
    c2w[:3, :3] = R_cw.T
    c2w[:3, 3] = -(R_cw.T @ t)
    return c2w


def colmap_intrinsics(cam: Dict) -> np.ndarray:
    """Build a 3×3 K matrix from a COLMAP camera info dictionary.

    Supports SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, and RADIAL models.
    Distortion parameters are ignored (COLMAP's undistorted images should
    be used for training).  All other camera models fall back to a rough
    FOV≈60° estimate.
    """
    model = cam["model"].upper()
    params = cam["params"]
    W, H = cam["width"], cam["height"]

    if model == "SIMPLE_PINHOLE":
        fx = fy = params[0]; cx, cy = params[1], params[2]
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    elif model in {"SIMPLE_RADIAL", "RADIAL"}:
        fx = fy = params[0]; cx, cy = params[1], params[2]
    else:                    # rough FOV ≈ 60° fallback
        fx = fy = max(W, H) * 0.866
        cx, cy = W / 2.0, H / 2.0

    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


# ---------------------------------------------------------------------------
# Standalone ray-generation helper (used by ColmapNeRFDataset and externally)
# ---------------------------------------------------------------------------

def get_rays(
    H: int,
    W: int,
    K: "torch.Tensor",   # (3, 3) intrinsic matrix
    c2w: "torch.Tensor", # (4, 4) camera-to-world matrix
) -> "Tuple[torch.Tensor, torch.Tensor]":
    """Return (rays_o, rays_d) tensors of shape (H, W, 3) for a pinhole camera."""
    device = K.device
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing="xy",
    )
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    dirs = torch.stack(
        [(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], dim=-1
    )  # (H, W, 3)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)  # (H, W, 3)
    rays_o = c2w[:3, 3].expand(rays_d.shape)                       # (H, W, 3)
    return rays_o, rays_d


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ColmapNeRFDataset:
    """Load a COLMAP binary reconstruction and matching RGB images for NeRF.

    Parameters
    ----------
    colmap_dir  : Sub-model folder containing ``cameras.bin``, ``images.bin``,
                  and ``points3D.bin``  (e.g. ``results/colmap_miniature/0``).
    image_dir   : Root directory where source images live.
    scale_factor: Resize images by this factor (<1 to downscale).  Smaller
                  images reduce memory and speed up training significantly.
    white_bg    : Composite over a white background during rendering.
    device      : ``"cpu"`` or ``"cuda"``.

    Attributes
    ----------
    images      : (N, H, W, 3) float32 tensor in [0, 1].
    c2w         : (N, 4, 4) float32 camera-to-world matrices (scene-normalised).
    K           : (N, 3, 3) float32 intrinsic matrices (adjusted for scale).
    near, far   : float – near/far plane distances in normalised scene units.
    scene_center: (3,) ndarray – world-space origin used for normalisation.
    scene_scale : float – world-space radius used for normalisation.
    rays_o      : (N·H·W, 3) float32 – pre-computed ray origins.
    rays_d      : (N·H·W, 3) float32 – pre-computed ray directions.
    targets     : (N·H·W, 3) float32 – ground-truth pixel colours.
    """

    def __init__(
        self,
        colmap_dir: str,
        image_dir: str,
        scale_factor: float = 1.0,
        white_bg: bool = True,
        device: str = "cpu",
    ) -> None:
        # Locate the colmap sibling module
        _here = Path(__file__).parent
        if str(_here) not in sys.path:
            sys.path.insert(0, str(_here))
        from colmap import load_colmap_bin_model  # type: ignore

        cams_dict, images_dict, points_dict = load_colmap_bin_model(colmap_dir)
        image_dir_p = Path(image_dir)

        sorted_ids = sorted(images_dict.keys(), key=lambda k: images_dict[k]["name"])
        c2w_list, K_list, img_list = [], [], []

        for img_id in sorted_ids:
            info = images_dict[img_id]
            img_path = self._locate_image(image_dir_p, info["name"])
            if img_path is None:
                print(f"[ColmapNeRFDataset] Warning: '{info['name']}' not found – skipping.")
                continue
            img = Image.open(img_path).convert("RGB")
            if scale_factor != 1.0:
                nw = max(1, int(img.width  * scale_factor))
                nh = max(1, int(img.height * scale_factor))
                img = img.resize((nw, nh), Image.LANCZOS)
            img_list.append(np.array(img, dtype=np.float32) / 255.0)

            cam = cams_dict[info["camera_id"]]
            K = colmap_intrinsics(cam)
            if scale_factor != 1.0:
                K = K.copy()
                K[0] *= scale_factor   # scale fx, cx
                K[1] *= scale_factor   # scale fy, cy
            K_list.append(K)
            c2w_list.append(colmap_to_c2w(info["qvec"], info["tvec"]))

        if not img_list:
            raise RuntimeError(
                "No images loaded.  Verify that colmap_dir and image_dir are correct."
            )

        c2w_arr = np.stack(c2w_list).astype(np.float32)   # (N, 4, 4)

        # ── Scene normalisation ──────────────────────────────────────────────
        cam_centers = c2w_arr[:, :3, 3]                   # (N, 3)
        self.scene_center = cam_centers.mean(axis=0)
        radii = np.linalg.norm(cam_centers - self.scene_center, axis=-1)
        self.scene_scale = float(radii.max()) or 1.0
        c2w_arr[:, :3, 3] = (c2w_arr[:, :3, 3] - self.scene_center) / self.scene_scale

        # ── Near/far from the sparse point cloud ────────────────────────────
        if points_dict:
            rng = np.random.default_rng(0)
            pts = np.stack([v["xyz"] for v in points_dict.values()]).astype(np.float32)
            pts = (pts - self.scene_center) / self.scene_scale
            if len(pts) > 50_000:                    # subsample for speed
                pts = pts[rng.choice(len(pts), 50_000, replace=False)]
            # Per-camera distances to all (subsampled) points.  (N, M)
            dists = np.linalg.norm(pts[None] - c2w_arr[:, :3, 3][:, None], axis=-1)
            self.near = float(max(np.percentile(dists, 0.5),  0.01))
            self.far  = float(np.percentile(dists, 99.5))
        else:
            self.near, self.far = 0.1, 6.0

        # ── Convert to tensors ───────────────────────────────────────────────
        self.c2w    = torch.from_numpy(c2w_arr).to(device)
        self.K      = torch.from_numpy(np.stack(K_list).astype(np.float32)).to(device)
        imgs_np     = np.stack(img_list)               # (N, H, W, 3)
        self.images = torch.from_numpy(imgs_np).to(device)
        self.white_bg = white_bg
        self.device   = device
        self.N, self.H, self.W = imgs_np.shape[:3]

        print(
            f"[ColmapNeRFDataset] {self.N} images ({self.H}×{self.W})  "
            f"near={self.near:.3f}  far={self.far:.3f}  "
            f"scene_scale={self.scene_scale:.4f}"
        )
        self._build_rays()

    # -------------------------------------------------------------------------
    @staticmethod
    def _locate_image(image_dir: Path, name: str) -> Optional[Path]:
        """Find an image file using several fallback strategies."""
        for candidate in [image_dir / name, image_dir / Path(name).name]:
            if candidate.exists():
                return candidate
        # Recursive search as last resort
        hits = list(image_dir.rglob(Path(name).name))
        return hits[0] if hits else None

    def _build_rays(self) -> None:
        """Pre-compute all ray origins, directions, and target colours."""
        all_o, all_d, all_t = [], [], []
        for i in range(self.N):
            ro, rd = get_rays(self.H, self.W, self.K[i], self.c2w[i])
            all_o.append(ro.reshape(-1, 3))
            all_d.append(rd.reshape(-1, 3))
            all_t.append(self.images[i].reshape(-1, 3))
        self.rays_o  = torch.cat(all_o, dim=0)   # (N·H·W, 3)
        self.rays_d  = torch.cat(all_d, dim=0)
        self.targets = torch.cat(all_t, dim=0)

    def sample_rays(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample *batch_size* rays from all training images.

        Returns
        -------
        rays_o  : (B, 3)
        rays_d  : (B, 3)
        targets : (B, 3) ground-truth RGB
        """
        idx = torch.randint(0, self.rays_o.shape[0], (batch_size,), device=self.device)
        return self.rays_o[idx], self.rays_d[idx], self.targets[idx]

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        """Return ``(image, c2w, K)`` for image *idx*."""
        return self.images[idx], self.c2w[idx], self.K[idx]

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def trans_t(t):
    return torch.tensor([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1]
    ], dtype=torch.float32)

def rot_phi(phi):
    return torch.tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]
    ], dtype=torch.float32)

def rot_theta(th):
    return torch.tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0,np.cos(th),0],
        [0,0,0,1]
    ], dtype=torch.float32)

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.tensor([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=torch.float32) @ c2w
    return c2w