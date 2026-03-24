"""
nerf.py – Fast NeRF building blocks for COLMAP-based Neural Radiance Fields.

Key components
--------------
PositionalEncoding  : Fourier feature embedding (Mildenhall et al. 2020).
TinyNeRF            : Compact all-in-one MLP, ideal for quick experiments.
FastNeRFModel       : Factored position/direction MLP (Garbin et al. 2021);
                      separates a position branch (density + radiance basis)
                      from a lightweight view-direction weighting branch,
                      enabling voxel-grid caching for ~200× faster rendering.
volume_render       : Differentiable alpha-compositing along a ray.
sample_stratified   : Jittered uniform sampling in [near, far].
sample_pdf          : Inverse-CDF importance sampling for hierarchical NeRF.
get_rays            : Build per-pixel ray bundles from camera intrinsics + c2w.
qvec_to_rotmat      : COLMAP quaternion → 3×3 rotation matrix.
colmap_to_c2w       : COLMAP (qvec, tvec) → 4×4 camera-to-world matrix.
colmap_intrinsics   : COLMAP camera dict → 3×3 K matrix.
ColmapNeRFDataset   : Loads images + COLMAP binary poses, normalises the scene,
                      pre-computes every ray, and exposes batched random sampling.
render_image        : Render a full H×W image from a trained model in chunks.

References
----------
* Mildenhall et al. "NeRF: Representing Scenes as Neural Radiance Fields
  for View Synthesis." ECCV 2020.
* Garbin et al. "FastNeRF: High-Fidelity Neural Rendering at 200FPS."
  ICCV 2021. https://arxiv.org/abs/2103.10380
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Fourier feature positional encoding (NeRF-style).

    Maps an input vector of dimension D to
    ``[x, sin(2⁰·x), cos(2⁰·x), …, sin(2^{L-1}·x), cos(2^{L-1}·x)]``
    where L = ``num_frequencies``.  Higher frequencies capture fine detail.
    """

    def __init__(self, num_frequencies: int = 10, include_input: bool = True) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        freqs = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer("freqs", freqs)

    def out_dim(self, input_dim: int = 3) -> int:
        d = 2 * self.num_frequencies * input_dim
        return d + input_dim if self.include_input else d

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (..., D) → (..., out_dim)
        parts = [x] if self.include_input else []
        for freq in self.freqs:
            parts.extend([torch.sin(freq * x), torch.cos(freq * x)])
        return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# TinyNeRF – compact all-in-one MLP (great for prototyping)
# ---------------------------------------------------------------------------

class TinyNeRF(nn.Module):
    """Compact NeRF MLP: position + view direction → (rgb, sigma).

    Faster to train than ``FastNeRFModel`` at the cost of peak rendering
    quality.  Recommended for sanity-checks and quick iteration.
    """

    def __init__(
        self,
        pos_freq: int = 6,
        dir_freq: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.pos_enc = PositionalEncoding(pos_freq)
        self.dir_enc = PositionalEncoding(dir_freq)
        pos_in = self.pos_enc.out_dim(3)
        dir_in = self.dir_enc.out_dim(3)

        layers: list[nn.Module] = []
        in_ch = pos_in
        for _ in range(num_layers):
            layers += [nn.Linear(in_ch, hidden_dim), nn.ReLU(inplace=True)]
            in_ch = hidden_dim
        self.pos_net = nn.Sequential(*layers)
        self.sigma_head = nn.Linear(hidden_dim, 1)
        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_dim + dir_in, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(
        self, pts: torch.Tensor, dirs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.pos_net(self.pos_enc(pts))
        sigma = F.softplus(self.sigma_head(feat))
        rgb = torch.sigmoid(
            self.rgb_head(torch.cat([feat, self.dir_enc(dirs)], dim=-1))
        )
        return rgb, sigma


# ---------------------------------------------------------------------------
# FastNeRFModel – factored position / direction branches (Garbin et al. 2021)
# ---------------------------------------------------------------------------

class FastNeRFModel(nn.Module):
    """Factored NeRF as in *FastNeRF: High-Fidelity Neural Rendering at 200FPS*
    (Garbin et al., ICCV 2021).

    Architecture
    ------------
    **Position branch** (deep MLP with skip connection):
        3-D sample point  →  volumetric density σ  +  K basis radiance vectors F ∈ ℝ^{K×3}

    **Direction branch** (shallow MLP, view-only):
        unit view direction  →  K scalar blend weights W ∈ ℝ^K  (softmax-normalised)

    Final colour: **c** = sigmoid(Σ_k W_k · F_k)

    The factorisation allows the position branch outputs to be baked into a
    voxel grid at inference time, reducing per-pixel cost to a cheap lookup
    plus one direction-MLP evaluation → ~200× speedup over vanilla NeRF.
    """

    def __init__(
        self,
        pos_freq: int = 10,
        dir_freq: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 6,
        basis_dim: int = 8,
        skip_layer: int = 3,
    ) -> None:
        super().__init__()
        self.pos_enc = PositionalEncoding(pos_freq)
        self.dir_enc = PositionalEncoding(dir_freq)
        self.basis_dim = basis_dim
        self.skip_layer = skip_layer

        pos_in = self.pos_enc.out_dim(3)
        dir_in = self.dir_enc.out_dim(3)

        # Position branch with one skip connection at layer `skip_layer`
        self.pos_layers = nn.ModuleList()
        in_ch = pos_in
        for i in range(num_layers):
            self.pos_layers.append(nn.Linear(in_ch, hidden_dim))
            in_ch = hidden_dim
            if i + 1 == skip_layer:          # next layer will receive the skip
                in_ch = hidden_dim + pos_in

        self.sigma_head = nn.Linear(hidden_dim, 1)
        self.basis_head = nn.Linear(hidden_dim, basis_dim * 3)  # K × RGB vectors

        # Direction branch (lightweight)
        self.dir_net = nn.Sequential(
            nn.Linear(dir_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, basis_dim),
        )

    def forward(
        self, pts: torch.Tensor, dirs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        pts  : (..., 3) world-space sample positions.
        dirs : (..., 3) **unit** view directions.

        Returns
        -------
        rgb   : (..., 3) colour in [0, 1].
        sigma : (..., 1) non-negative volumetric density.
        """
        pos_enc = self.pos_enc(pts)
        h = pos_enc
        for i, layer in enumerate(self.pos_layers):
            h = F.relu(layer(h))
            if i + 1 == self.skip_layer:
                h = torch.cat([h, pos_enc], dim=-1)

        sigma = F.softplus(self.sigma_head(h))                          # (..., 1)
        basis = self.basis_head(h).view(*pts.shape[:-1], self.basis_dim, 3)  # (..., K, 3)
        weights = torch.softmax(self.dir_net(self.dir_enc(dirs)), dim=-1)    # (..., K)
        rgb = torch.sigmoid((weights.unsqueeze(-1) * basis).sum(dim=-2))     # (..., 3)
        return rgb, sigma


# ---------------------------------------------------------------------------
# Volume rendering
# ---------------------------------------------------------------------------

def volume_render(
    rgb: torch.Tensor,
    sigma: torch.Tensor,
    t_vals: torch.Tensor,
    rays_d: torch.Tensor,
    white_bg: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable alpha compositing (numerical quadrature along a ray).

    Parameters
    ----------
    rgb    : (R, S, 3)  per-sample colour.
    sigma  : (R, S, 1)  per-sample density (non-negative).
    t_vals : (R, S)     distance of each sample from the ray origin.
    rays_d : (R, 3)     ray direction vectors (need not be unit vectors;
                        their norm is used to convert t-steps to world-space).
    white_bg : composite over a white background if True.

    Returns
    -------
    colour  : (R, 3)
    depth   : (R,)
    weights : (R, S)   useful for importance sampling.
    """
    # Step sizes in t-space; treat the last step as infinite
    dists = t_vals[..., 1:] - t_vals[..., :-1]                              # (R, S-1)
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)  # (R, S)
    # Scale by ray magnitude → actual world-space step lengths
    dists = dists * rays_d.norm(dim=-1, keepdim=True)                       # (R, S)

    alpha = 1.0 - torch.exp(-F.relu(sigma[..., 0]) * dists)                # (R, S)

    # Transmittance: T_i = ∏_{j<i}(1 – α_j)
    T = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[..., :-1]                                                             # (R, S)

    weights = T * alpha                                                     # (R, S)
    colour = (weights.unsqueeze(-1) * rgb).sum(dim=-2)                     # (R, 3)
    depth = (weights * t_vals).sum(dim=-1)                                 # (R,)

    if white_bg:
        colour = colour + (1.0 - weights.sum(dim=-1, keepdim=True))

    return colour, depth, weights


# ---------------------------------------------------------------------------
# Ray sampling
# ---------------------------------------------------------------------------

def sample_stratified(
    near: float,
    far: float,
    n_samples: int,
    n_rays: int,
    device: torch.device,
    perturb: bool = True,
) -> torch.Tensor:
    """Jittered stratified sampling in [near, far].

    Returns
    -------
    t_vals : (n_rays, n_samples)
    """
    t = torch.linspace(near, far, n_samples, device=device).expand(n_rays, n_samples)
    if perturb:
        mid = 0.5 * (t[..., 1:] + t[..., :-1])
        upper = torch.cat([mid, t[..., -1:]], dim=-1)
        lower = torch.cat([t[..., :1], mid], dim=-1)
        t = lower + (upper - lower) * torch.rand_like(t)
    return t


def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
    perturb: bool = True,
) -> torch.Tensor:
    """Inverse-CDF importance sampling for hierarchical NeRF.

    Parameters
    ----------
    bins      : (R, W) – midpoints of W bins.  Typically the midpoints of
                consecutive coarse t-values, smoothed with neighbour weights.
    weights   : (R, W) – un-normalised non-negative importance weights.
                Must have the **same** last dimension as ``bins``.
    n_samples : number of fine samples to draw per ray.

    Returns
    -------
    samples : (R, n_samples) new t-values drawn from the implied distribution.

    Notes
    -----
    Linear interpolation between neighbouring bin centres is used within each
    CDF interval, matching the standard nerf-pytorch convention.
    """
    w = weights.clamp(min=0.0) + 1e-5
    pdf = w / w.sum(dim=-1, keepdim=True)                                # (R, W)
    cdf = torch.cat(
        [torch.zeros_like(pdf[..., :1]), torch.cumsum(pdf, dim=-1)],
        dim=-1,
    )                                                                    # (R, W+1)

    if perturb:
        u = torch.rand(*bins.shape[:-1], n_samples, device=bins.device)
    else:
        u = torch.linspace(0.0, 1.0, n_samples, device=bins.device).expand(
            *bins.shape[:-1], n_samples
        )
    u = u.contiguous()

    # Binary search: for each u find the interval.  inds ∈ [0, W].
    inds = torch.searchsorted(cdf, u, right=True)                       # (R, F)

    # Left index ∈ [0, W-2], right = left+1 ∈ [1, W-1]
    lo = (inds - 1).clamp(0, bins.shape[-1] - 2)                       # (R, F)
    hi = lo + 1                                                         # (R, F)

    bins_lo = torch.gather(bins, -1, lo)                                # (R, F)
    bins_hi = torch.gather(bins, -1, hi)                                # (R, F)
    cdf_lo  = torch.gather(cdf,  -1, lo)                               # (R, F)
    cdf_hi  = torch.gather(cdf,  -1, hi)                               # (R, F)

    denom = (cdf_hi - cdf_lo).clamp(min=1e-5)
    t = ((u - cdf_lo) / denom).clamp(0.0, 1.0)
    return bins_lo + t * (bins_hi - bins_lo)


# ---------------------------------------------------------------------------
# Ray generation
# ---------------------------------------------------------------------------

def get_rays(
    H: int,
    W: int,
    K: torch.Tensor,
    c2w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate per-pixel rays for one image.

    Parameters
    ----------
    H, W : image height and width.
    K    : (3, 3) intrinsic matrix  ``[fx, 0, cx; 0, fy, cy; 0, 0, 1]``.
    c2w  : (4, 4) camera-to-world matrix produced by ``colmap_to_c2w``.

    Returns
    -------
    rays_o : (H, W, 3) ray origins  (= camera centre, tiled over all pixels).
    rays_d : (H, W, 3) ray directions in world space (not unit-normalised).

    Notes
    -----
    COLMAP uses +X right, +Y down, +Z into the scene.  NeRF / OpenGL uses
    +X right, +Y **up**, −Z into the scene.  The conversion is applied here
    by negating the Y and Z pixel-direction components before rotating to
    world space, so the rest of the pipeline uses NeRF conventions throughout.
    """
    device = c2w.device
    j, i = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    # Pixel directions in COLMAP camera space  (+Z forward)
    dx = (i - K[0, 2]) / K[0, 0]
    dy = (j - K[1, 2]) / K[1, 1]
    # Flip Y and Z → NeRF/OpenGL convention  (−Z forward, +Y up)
    dirs = torch.stack([dx, -dy, -torch.ones_like(dx)], dim=-1)         # (H, W, 3)

    # Rotate to world space
    rays_d = (dirs[..., None, :] * c2w[:3, :3]).sum(dim=-1)            # (H, W, 3)
    rays_o = c2w[:3, 3].expand_as(rays_d)                              # (H, W, 3)
    return rays_o, rays_d


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


# ---------------------------------------------------------------------------
# Full-image rendering
# ---------------------------------------------------------------------------

@torch.no_grad()
def render_image(
    model: nn.Module,
    H: int,
    W: int,
    K: torch.Tensor,
    c2w: torch.Tensor,
    near: float,
    far: float,
    n_coarse: int = 64,
    n_fine: int = 64,
    chunk: int = 2048,
    white_bg: bool = True,
    device: str = "cpu",
    coarse_model: Optional[nn.Module] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render a full H×W image from a trained NeRF model.

    Processes rays in batches of *chunk* to prevent OOM errors.

    Parameters
    ----------
    model        : Fine (or only) NeRF model.
    coarse_model : When provided, hierarchical importance sampling is used:
                   coarse weights guide fine sample placement for higher quality.
    n_coarse     : Number of stratified samples per ray (coarse pass).
    n_fine       : Number of importance-sampled points added in the fine pass
                   (only used when ``coarse_model`` is not None).
    chunk        : Number of rays processed in one forward pass.

    Returns
    -------
    rgb_img   : (H, W, 3) float32 in [0, 1].
    depth_img : (H, W)    float32 (normalised scene units).
    """
    model.eval()
    if coarse_model is not None:
        coarse_model.eval()

    rays_o, rays_d = get_rays(H, W, K, c2w.to(device))
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    rgb_chunks, depth_chunks = [], []
    for start in range(0, rays_o.shape[0], chunk):
        ro = rays_o[start : start + chunk]
        rd = rays_d[start : start + chunk]
        B  = ro.shape[0]

        # ── Coarse pass ──────────────────────────────────────────────────────
        t_c  = sample_stratified(near, far, n_coarse, B, device=device)  # (B, Sc)
        pts  = ro[:, None] + rd[:, None] * t_c[..., None]                # (B, Sc, 3)
        d_n  = F.normalize(rd, dim=-1)[:, None].expand_as(pts)           # (B, Sc, 3)
        src  = coarse_model if coarse_model is not None else model
        rgb_c, sig_c = src(pts.reshape(-1, 3), d_n.reshape(-1, 3))
        rgb_c = rgb_c.reshape(B, n_coarse, 3)
        sig_c = sig_c.reshape(B, n_coarse, 1)

        if coarse_model is not None and n_fine > 0:
            # ── Fine pass (hierarchical importance sampling) ─────────────────
            _, _, w_c = volume_render(rgb_c, sig_c, t_c, rd, white_bg=white_bg)
            t_mid  = 0.5 * (t_c[..., 1:] + t_c[..., :-1])               # (B, Sc-1)
            w_mid  = 0.5 * (w_c[..., :-1] + w_c[..., 1:])               # (B, Sc-1)
            t_f    = sample_pdf(t_mid, w_mid, n_fine, perturb=False)     # (B, Sf)
            t_all, _ = torch.sort(torch.cat([t_c, t_f], dim=-1), dim=-1)
            pts_f  = ro[:, None] + rd[:, None] * t_all[..., None]
            d_nf   = F.normalize(rd, dim=-1)[:, None].expand_as(pts_f)
            rgb_c, sig_c = model(pts_f.reshape(-1, 3), d_nf.reshape(-1, 3))
            rgb_c  = rgb_c.reshape(B, -1, 3)
            sig_c  = sig_c.reshape(B, -1, 1)
            t_c    = t_all

        colour, depth, _ = volume_render(rgb_c, sig_c, t_c, rd, white_bg=white_bg)
        rgb_chunks.append(colour.cpu())
        depth_chunks.append(depth.cpu())

    rgb_img   = torch.cat(rgb_chunks).reshape(H, W, 3).numpy()
    depth_img = torch.cat(depth_chunks).reshape(H, W).numpy()
    return rgb_img.clip(0.0, 1.0).astype(np.float32), depth_img.astype(np.float32)
