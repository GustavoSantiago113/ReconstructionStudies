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
# Ray Batching
# ---------------------------------------------------------------------------
def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0],
                     -(j-K[1][2])/K[1][1],
                     -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def get_rays(H: int, W: int, K, c2w: torch.Tensor):
    """Generate rays for every pixel (PyTorch version).

    Parameters
    ----------
    K : torch.Tensor (3, 3) intrinsic matrix **or** scalar focal length.
    c2w : torch.Tensor (4, 4) or (3, 4) camera-to-world matrix.
    """
    device = c2w.device
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing='xy',
    )
    if isinstance(K, torch.Tensor) and K.dim() >= 2:
        dirs = torch.stack([
            (i - K[0, 2]) / K[0, 0],
            -(j - K[1, 2]) / K[1, 1],
            -torch.ones_like(i),
        ], dim=-1)
    else:
        focal = float(K) if not isinstance(K, torch.Tensor) else K.item()
        dirs = torch.stack([
            (i - W * 0.5) / focal,
            -(j - H * 0.5) / focal,
            -torch.ones_like(i),
        ], dim=-1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_o = c2w[:3, 3].expand_as(rays_d)
    return rays_o, rays_d


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
         
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
             
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
         
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
             
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                     
        self.embed_fns = embed_fns
        self.out_dim = out_dim
         
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
 
def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
     
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
     
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

def pos_enc(x, L_embed=6):
    rets = [x]
    for i in range(L_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2.**i * x))
    return torch.cat(rets, dim=-1)

# ------------------------------------------------------------------------------
# NeRF MLP
# ------------------------------------------------------------------------------

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
 
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)]
        )
 
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
 
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
 
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
 
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
 
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
 
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
 
        return outputs

# ---------------------------------------------------------------------------
# Instant-NGP: Multiresolution Hash Encoding  (Müller et al. 2022)
# ---------------------------------------------------------------------------

def sh_encode_directions(d: torch.Tensor) -> torch.Tensor:
    """Third-order real spherical harmonics (16 coefficients, degree 0–3).

    Parameters
    ----------
    d : (..., 3) unit direction vectors.

    Returns
    -------
    (..., 16) SH features.
    """
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    return torch.stack([
        # degree 0
        0.28209479177387814 * torch.ones_like(x),
        # degree 1
        -0.4886025119029199  * y,
         0.4886025119029199  * z,
        -0.4886025119029199  * x,
        # degree 2
         1.0925484305920792  * xy,
        -1.0925484305920792  * yz,
         0.31539156525252005 * (2.0 * zz - xx - yy),
        -1.0925484305920792  * xz,
         0.5462742152960396  * (xx - yy),
        # degree 3
        -0.5900435899266435  * y * (3.0 * xx - yy),
         2.890611442640554   * xy * z,
        -0.4570457994644658  * y * (4.0 * zz - xx - yy),
         0.3731763325901154  * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
        -0.4570457994644658  * x * (4.0 * zz - xx - yy),
         1.445305721320277   * z * (xx - yy),
        -0.5900435899266435  * x * (xx - 3.0 * yy),
    ], dim=-1)


class MultiresHashEncoding(nn.Module):
    """Multiresolution hash grid encoding from Instant-NGP.

    Parameters
    ----------
    n_levels            : Number of resolution levels L (default 16).
    n_features_per_level: Feature dimensionality per level F (default 2).
    log2_hashmap_size   : log₂ of the per-level hash table size T (default 19 → 524 288 entries).
    base_resolution     : Coarsest grid resolution N_min (default 16).
    max_resolution      : Finest grid resolution N_max (default 2 048).
    """

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        max_resolution: int = 2048,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.hashmap_size = 2 ** log2_hashmap_size
        self.out_dim = n_levels * n_features_per_level

        b = np.exp(np.log(max_resolution / base_resolution) / (n_levels - 1))
        resolutions = [int(np.floor(base_resolution * (b ** l))) for l in range(n_levels)]
        self.register_buffer("resolutions", torch.tensor(resolutions, dtype=torch.float32))

        # All 8 trilinear corners as a fixed (8, 3) buffer — avoids realloc each forward
        corners = torch.tensor(
            [[dx, dy, dz] for dx in range(2) for dy in range(2) for dz in range(2)],
            dtype=torch.long,
        )  # (8, 3)
        self.register_buffer("corners_offsets", corners)

        # Single stacked embedding (L*T, F) — enables one vectorised lookup for all levels
        self.stacked_emb = nn.Parameter(
            torch.empty(n_levels * self.hashmap_size, n_features_per_level).uniform_(-1e-4, 1e-4)
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """Hash-encode a batch of 3-D positions.

        Parameters
        ----------
        xyz : (..., 3) positions in normalised scene space (expected range ≈ [−1, 1]).

        Returns
        -------
        (..., out_dim) concatenated multi-resolution features.
        """
        L = self.n_levels
        T = self.hashmap_size
        F = self.n_features_per_level

        original_shape = xyz.shape[:-1]
        xyz_flat = xyz.reshape(-1, 3)                          # (N, 3)
        N = xyz_flat.shape[0]
        xyz_01 = xyz_flat.clamp(-1.0, 1.0) * 0.5 + 0.5       # (N, 3) → [0, 1]

        # Scale to each level's resolution: (N, L, 3)
        res = self.resolutions  # (L,) float
        scaled = xyz_01.unsqueeze(1) * (res.unsqueeze(-1) - 1.0)  # (N, L, 3)
        xi = scaled.floor().long()    # (N, L, 3) lower-left corner
        xf = scaled - xi.float()      # (N, L, 3) fractional weights

        # All 8 corners for every point and every level: (N, L, 8, 3)
        corners = xi.unsqueeze(2) + self.corners_offsets      # (N, L, 8, 3)

        # Spatial hash: XOR of prime-multiplied coordinates, result in [0, T-1]
        h = corners[..., 0]
        h = h ^ (corners[..., 1] * 2_654_435_761)
        h = h ^ (corners[..., 2] * 805_459_861)
        hash_idx = h & (T - 1)                                # (N, L, 8)

        # Offset by level so each level addresses its own slice of stacked_emb
        level_offsets = torch.arange(L, device=xyz.device, dtype=torch.long) * T  # (L,)
        hash_idx = hash_idx + level_offsets.unsqueeze(0).unsqueeze(-1)             # (N, L, 8)

        # Single lookup for all points, levels, and corners (N*L*8, F) → (N, L, 8, F)
        feat = self.stacked_emb[hash_idx.reshape(-1)].reshape(N, L, 8, F)

        # Trilinear weights — outer product over the three axes: (N, L, 2, 2, 2) → (N, L, 8)
        wx = torch.stack([1.0 - xf[..., 0], xf[..., 0]], dim=-1)  # (N, L, 2)
        wy = torch.stack([1.0 - xf[..., 1], xf[..., 1]], dim=-1)
        wz = torch.stack([1.0 - xf[..., 2], xf[..., 2]], dim=-1)
        w = wx[..., :, None, None] * wy[..., None, :, None] * wz[..., None, None, :]  # (N, L, 2, 2, 2)
        w = w.reshape(N, L, 8)                                     # (N, L, 8)

        # Weighted sum over the 8 corners: (N, L, F)
        feat_interp = (w.unsqueeze(-1) * feat).sum(dim=2)         # (N, L, F)

        return feat_interp.reshape(N, L * F).reshape(*original_shape, self.out_dim)


class InstantNGP(nn.Module):
    """Instant-NGP radiance field  (Müller et al. 2022).

    Drop-in replacement for :class:`NeRF`.  The model owns its multiresolution
    hash encoding and a compact two-hidden-layer MLP, so **no external
    positional-encoding functions are needed**.

    Usage with :func:`run_network`
    ------------------------------
    Pass ``embed_fn=None`` and ``embeddirs_fn=(lambda x: x)`` so that raw
    positions (first 3 channels) and raw view directions (last 3 channels) are
    concatenated and forwarded to this model unchanged.

    Parameters
    ----------
    n_levels            : Hash-grid levels L (default 16).
    n_features_per_level: Features per level F (default 2 → 32-dim hash output).
    log2_hashmap_size   : log₂ hash table size per level T (default 19).
    base_resolution     : Coarsest grid resolution N_min (default 16).
    max_resolution      : Finest grid resolution N_max (default 2 048).
    geo_hidden          : Hidden width of the geometry MLP (default 64).
    color_hidden        : Hidden width of the colour MLP (default 64).
    use_viewdirs        : Condition colour on view direction via 3rd-order SH if True.
    """

    SH_DIM: int = 16   # (degree 0–3): 1 + 3 + 5 + 7 = 16 coefficients

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        max_resolution: int = 2048,
        geo_hidden: int = 64,
        color_hidden: int = 64,
        use_viewdirs: bool = True,
    ):
        super().__init__()
        self.use_viewdirs = use_viewdirs

        self.hash_enc = MultiresHashEncoding(
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            max_resolution=max_resolution,
        )
        hash_dim = self.hash_enc.out_dim   # n_levels * n_features_per_level

        # Geometry network: hash features → raw density + bottleneck feature
        self.geo_net = nn.Sequential(
            nn.Linear(hash_dim, geo_hidden),
            nn.ReLU(),
            nn.Linear(geo_hidden, 1 + geo_hidden),
        )

        # Colour network: bottleneck (+ optional SH dirs) → raw RGB
        dir_dim = self.SH_DIM if use_viewdirs else 0
        self.color_net = nn.Sequential(
            nn.Linear(geo_hidden + dir_dim, color_hidden),
            nn.ReLU(),
            nn.Linear(color_hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the radiance field.

        Parameters
        ----------
        x : (N, 3) raw positions, or (N, 6) positions + view directions.
            Positions occupy the first 3 channels; unit view directions the
            last 3 channels (only used when ``use_viewdirs=True``).

        Returns
        -------
        (N, 4) tensor ``[R, G, B, density_logit]`` – identical layout to
        :class:`NeRF`, compatible with :func:`raw2outputs`.
        """
        pts  = x[..., :3]
        dirs = x[..., 3:6] if self.use_viewdirs else None

        # Hash-encode positions
        h = self.hash_enc(pts)                                # (N, hash_dim)

        # Geometry head: raw density + bottleneck feature
        geo_out = self.geo_net(h)                             # (N, 1 + geo_hidden)
        sigma   = geo_out[..., :1]                            # raw density logit
        feat    = geo_out[..., 1:]                            # geometry feature

        # Colour head
        if self.use_viewdirs and dirs is not None:
            sh       = sh_encode_directions(F.normalize(dirs, dim=-1))   # (N, 16)
            color_in = torch.cat([feat, sh], dim=-1)
        else:
            color_in = feat

        rgb = self.color_net(color_in)                        # (N, 3)
        return torch.cat([rgb, sigma], dim=-1)                # (N, 4)


def get_instant_ngp(
    use_viewdirs: bool = True, **kwargs
) -> Tuple["InstantNGP", None, Optional[object]]:
    """Create an :class:`InstantNGP` model with matching encode functions.

    Returns
    -------
    model        : :class:`InstantNGP` instance.
    embed_fn     : ``None`` – positions are encoded inside the model.
    embeddirs_fn : identity ``lambda`` (for raw view dirs) if ``use_viewdirs``, else ``None``.
    """
    model = InstantNGP(use_viewdirs=use_viewdirs, **kwargs)
    embeddirs_fn = (lambda x: x) if use_viewdirs else None
    return model, None, embeddirs_fn


# ---------------------------------------------------------------------------
# FastNeRF  (Garbin et al., ICCV 2021)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with optional raw-input pass-through."""

    def __init__(self, multires: int, include_input: bool = True) -> None:
        super().__init__()
        self.multires = multires
        self.include_input = include_input

    def out_dim(self, input_dim: int) -> int:
        base = input_dim if self.include_input else 0
        return base + input_dim * 2 * self.multires

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rets = [x] if self.include_input else []
        for i in range(self.multires):
            rets += [torch.sin(2.0 ** i * x), torch.cos(2.0 ** i * x)]
        return torch.cat(rets, dim=-1)


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

    Usage with :func:`run_network`
    ------------------------------
    Pass ``embed_fn=None`` and ``embeddirs_fn=(lambda x: x)`` — identical to
    :class:`InstantNGP`.  ``run_network`` will concatenate raw xyz + view dirs
    into a 6-channel vector which this model unpacks internally.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the radiance field.

        Parameters
        ----------
        x : (N, 3) raw positions, or (N, 6) positions + unit view directions.
            Layout matches :class:`InstantNGP` for compatibility with
            :func:`run_network` when ``embed_fn=None``.

        Returns
        -------
        (N, 4) tensor ``[R, G, B, density]`` – identical layout to
        :class:`NeRF`, compatible with :func:`raw2outputs`.
        Note: RGB channels are **pre-sigmoid logits**; ``raw2outputs`` applies
        sigmoid itself.
        """
        pts  = x[..., :3]
        dirs = F.normalize(x[..., 3:6], dim=-1) if x.shape[-1] >= 6 else None

        pos_enc = self.pos_enc(pts)
        h = pos_enc
        for i, layer in enumerate(self.pos_layers):
            h = F.relu(layer(h))
            if i + 1 == self.skip_layer:
                h = torch.cat([h, pos_enc], dim=-1)

        sigma = F.softplus(self.sigma_head(h))                               # (N, 1)
        basis = self.basis_head(h).view(*pts.shape[:-1], self.basis_dim, 3)  # (N, K, 3)

        if dirs is not None:
            weights = torch.softmax(self.dir_net(self.dir_enc(dirs)), dim=-1)  # (N, K)
        else:
            weights = torch.full((*pts.shape[:-1], self.basis_dim), 1.0 / self.basis_dim,
                                 device=pts.device, dtype=pts.dtype)

        # Pre-sigmoid logits — raw2outputs applies sigmoid to raw[..., :3]
        rgb_logit = (weights.unsqueeze(-1) * basis).sum(dim=-2)              # (N, 3)
        return torch.cat([rgb_logit, sigma], dim=-1)                         # (N, 4)


def get_fast_nerf(use_viewdirs: bool = True, **kwargs) -> Tuple["FastNeRFModel", None, Optional[object]]:
    """Create a :class:`FastNeRFModel` with matching encode functions.

    Returns
    -------
    model        : :class:`FastNeRFModel` instance.
    embed_fn     : ``None`` – positions are encoded inside the model.
    embeddirs_fn : identity ``lambda`` (for raw view dirs) if ``use_viewdirs``, else ``None``.
    """
    model = FastNeRFModel(**kwargs)
    embeddirs_fn = (lambda x: x) if use_viewdirs else None
    return model, None, embeddirs_fn


# ---------------------------------------------------------------------------
# Hierarchical sampling
# ---------------------------------------------------------------------------

def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # A zero is prepended to the CDF to handle boundary conditions, ensuring the CDF starts from 0 and ends at 1.
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))
 
    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device)
 
    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)
 
    # Inver CDF
    u = u.contiguous() # ensures that `u` has contiguous memory layout
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
 
    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
 
    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
 
    return samples

# ---------------------------------------------------------------------------
# Volume rendering
# ---------------------------------------------------------------------------

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
 
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.tensor([1e10], device=z_vals.device).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
 
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
 
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape, device=raw.device) * raw_noise_std
 
        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)
 
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
 
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)
 
    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])
 
    return rgb_map, disp_map, acc_map, weights, depth_map


# ---------------------------------------------------------------------------
# Network evaluation & volumetric rendering
# ---------------------------------------------------------------------------

def run_network(model, pts, viewdirs, embed_fn, embeddirs_fn, netchunk=65536):
    """Evaluate the NeRF MLP on a batch of 3-D points with view directions.

    Parameters
    ----------
    model       : NeRF module.
    pts         : (N_rays, N_samples, 3) sampled points.
    viewdirs    : (N_rays, 3) unit view directions.
    embed_fn    : positional-encoding function for xyz.
    embeddirs_fn: positional-encoding function for view dirs.
    netchunk    : max points per forward pass (controls GPU memory).
    """
    pts_flat = pts.reshape(-1, 3)
    embedded = embed_fn(pts_flat) if embed_fn is not None else pts_flat

    if embeddirs_fn is not None:
        dirs_flat = viewdirs[:, None, :].expand_as(pts).reshape(-1, 3)
        embedded_dirs = embeddirs_fn(dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], dim=-1)

    outputs = []
    for i in range(0, embedded.shape[0], netchunk):
        outputs.append(model(embedded[i:i + netchunk]))
    raw = torch.cat(outputs, dim=0)
    return raw.reshape(list(pts.shape[:-1]) + [raw.shape[-1]])


def render_rays(
    model,
    rays_o,
    rays_d,
    near,
    far,
    N_samples,
    N_importance=0,
    model_fine=None,
    embed_fn=None,
    embeddirs_fn=None,
    rand=True,
    white_bkgd=False,
    raw_noise_std=0.0,
    netchunk=65536,
):
    """Full volumetric rendering pipeline (coarse + optional hierarchical fine).

    Returns
    -------
    dict with keys:
        rgb_map   – (N_rays, 3)
        depth_map – (N_rays,)
        acc_map   – (N_rays,)
        disp_map  – (N_rays,)
    If *N_importance* > 0 the dict also contains ``rgb_map_coarse``.
    """
    N_rays = rays_o.shape[0]
    device = rays_o.device

    # ---- stratified sampling along each ray --------------------------------
    t_vals = torch.linspace(0.0, 1.0, steps=N_samples, device=device)
    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals.expand(N_rays, N_samples)

    if rand:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        z_vals = lower + (upper - lower) * torch.rand(z_vals.shape, device=device)

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    viewdirs = F.normalize(rays_d, dim=-1)

    # ---- coarse network ----------------------------------------------------
    raw = run_network(model, pts, viewdirs, embed_fn, embeddirs_fn, netchunk)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd,
    )

    ret = {
        "rgb_map": rgb_map,
        "disp_map": disp_map,
        "acc_map": acc_map,
        "depth_map": depth_map,
    }

    # ---- hierarchical (fine) sampling --------------------------------------
    if N_importance > 0:
        fine_model = model_fine if model_fine is not None else model

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(not rand),
        )
        z_samples = z_samples.detach()
        z_vals_fine, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)

        pts_fine = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_fine[..., :, None]
        raw_fine = run_network(fine_model, pts_fine, viewdirs, embed_fn, embeddirs_fn, netchunk)
        rgb_fine, disp_fine, acc_fine, _, depth_fine = raw2outputs(
            raw_fine, z_vals_fine, rays_d, raw_noise_std, white_bkgd,
        )

        ret["rgb_map_coarse"] = rgb_map
        ret["rgb_map"] = rgb_fine
        ret["disp_map"] = disp_fine
        ret["acc_map"] = acc_fine
        ret["depth_map"] = depth_fine

    return ret


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