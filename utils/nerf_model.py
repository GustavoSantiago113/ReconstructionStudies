import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """Positional encoding for input coordinates"""
    def __init__(self, num_freqs=10, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        
    def forward(self, x):
        """
        Args:
            x: [..., dim] input tensor
        Returns:
            [..., dim * (2 * num_freqs + include_input)] encoded tensor
        """
        out = []
        if self.include_input:
            out.append(x)
        
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        
        return torch.cat(out, dim=-1)


class NeRFModel(nn.Module):
    """Neural Radiance Field model"""
    def __init__(
        self,
        pos_freq=10,
        dir_freq=4,
        hidden_dim=256,
        num_layers=8,
        skip_layer=4
    ):
        super().__init__()
        
        self.pos_encoding = PositionalEncoding(num_freqs=pos_freq, include_input=True)
        self.dir_encoding = PositionalEncoding(num_freqs=dir_freq, include_input=True)
        
        # Calculate input dimensions
        pos_input_dim = 3 * (2 * pos_freq + 1)  # 3 coords * (2 * freqs + original)
        dir_input_dim = 3 * (2 * dir_freq + 1)
        
        # Position network (processes 3D position)
        self.pos_layers = nn.ModuleList()
        self.pos_layers.append(nn.Linear(pos_input_dim, hidden_dim))
        
        for i in range(1, num_layers):
            if i == skip_layer:
                self.pos_layers.append(nn.Linear(hidden_dim + pos_input_dim, hidden_dim))
            else:
                self.pos_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Density head (sigma)
        self.density_head = nn.Linear(hidden_dim, 1)
        
        # Feature vector for color prediction
        self.feature_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # Direction network (processes view direction + features)
        self.dir_layer = nn.Linear(hidden_dim + dir_input_dim, hidden_dim // 2)
        
        # Color head (RGB)
        self.color_head = nn.Linear(hidden_dim // 2, 3)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, positions, directions):
        """
        Args:
            positions: [N, 3] 3D positions
            directions: [N, 3] view directions
        Returns:
            rgb: [N, 3] RGB colors
            sigma: [N, 1] volume densities
        """
        # Encode inputs
        pos_encoded = self.pos_encoding(positions)
        dir_encoded = self.dir_encoding(directions)
        
        # Process position through network
        x = pos_encoded
        for i, layer in enumerate(self.pos_layers):
            if i == 4:  # Skip connection
                x = torch.cat([x, pos_encoded], dim=-1)
            x = self.relu(layer(x))
        
        # Predict density
        sigma = self.relu(self.density_head(x))
        
        # Get features for color prediction
        features = self.feature_linear(x)
        
        # Combine features with view direction
        x = torch.cat([features, dir_encoded], dim=-1)
        x = self.relu(self.dir_layer(x))
        
        # Predict color
        rgb = self.sigmoid(self.color_head(x))
        
        return rgb, sigma


def volume_rendering(rgb, sigma, z_vals, directions, white_background=False):
    """
    Volumetric rendering using classical volume rendering equation
    
    Args:
        rgb: [N_rays, N_samples, 3] RGB values
        sigma: [N_rays, N_samples, 1] density values
        z_vals: [N_rays, N_samples] depth values along rays
        directions: [N_rays, 3] ray directions (for distance calculation)
        white_background: whether to use white background
        
    Returns:
        rgb_map: [N_rays, 3] rendered RGB
        depth_map: [N_rays] rendered depth
        acc_map: [N_rays] accumulated opacity
        weights: [N_rays, N_samples] rendering weights
    """
    # Calculate distances between samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
    dists = torch.clamp(dists, min=1e-8)  # Prevent zero distances
    
    # Multiply by ray direction norm to get real distance
    ray_norms = torch.norm(directions[..., None, :], dim=-1)
    ray_norms = torch.clamp(ray_norms, min=1e-8)  # Prevent zero norms
    dists = dists * ray_norms
    
    # Clamp sigma to prevent overflow
    sigma_clamped = torch.clamp(sigma.squeeze(-1), min=0.0, max=50.0)
    
    # Calculate alpha values (opacity) with numerical stability
    alpha_raw = sigma_clamped * dists
    alpha_raw = torch.clamp(alpha_raw, max=20.0)  # Prevent exp overflow
    alpha = 1.0 - torch.exp(-alpha_raw)
    alpha = torch.clamp(alpha, 0.0, 1.0)
    
    # Calculate transmittance (accumulated transparency) with numerical stability
    one_minus_alpha = torch.clamp(1.0 - alpha, min=1e-10, max=1.0)
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(one_minus_alpha[..., :1]), one_minus_alpha], dim=-1),
        dim=-1
    )[..., :-1]
    transmittance = torch.clamp(transmittance, min=1e-10, max=1.0)
    
    # Calculate rendering weights
    weights = alpha * transmittance
    weights = torch.clamp(weights, min=0.0, max=1.0)
    
    # Clamp RGB to valid range
    rgb_clamped = torch.clamp(rgb, 0.0, 1.0)
    
    # Render RGB
    rgb_map = torch.sum(weights[..., None] * rgb_clamped, dim=-2)
    rgb_map = torch.clamp(rgb_map, 0.0, 1.0)
    
    # Render depth
    depth_map = torch.sum(weights * z_vals, dim=-1)
    
    # Calculate accumulated opacity
    acc_map = torch.sum(weights, dim=-1)
    
    # Add white background if specified
    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])
    
    return rgb_map, depth_map, acc_map, weights


def sample_along_rays(ray_origins, ray_directions, near, far, num_samples, perturb=True):
    """
    Sample points along rays
    
    Args:
        ray_origins: [N_rays, 3] ray origins
        ray_directions: [N_rays, 3] ray directions (normalized)
        near: float or [N_rays] near bounds
        far: float or [N_rays] far bounds
        num_samples: number of samples per ray
        perturb: whether to add random perturbation
        
    Returns:
        points: [N_rays, N_samples, 3] sampled 3D points
        z_vals: [N_rays, N_samples] depth values
    """
    # Create evenly spaced samples
    t_vals = torch.linspace(0.0, 1.0, num_samples, device=ray_origins.device)
    
    if not isinstance(near, torch.Tensor):
        near = torch.full((ray_origins.shape[0],), near, device=ray_origins.device)
    if not isinstance(far, torch.Tensor):
        far = torch.full((ray_origins.shape[0],), far, device=ray_origins.device)
    
    z_vals = near[:, None] * (1.0 - t_vals[None, :]) + far[:, None] * t_vals[None, :]
    
    # Add random perturbation for training
    if perturb:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand
    
    # Calculate 3D points
    points = ray_origins[:, None, :] + ray_directions[:, None, :] * z_vals[..., None]
    
    return points, z_vals


def hierarchical_sampling(ray_origins, ray_directions, z_vals, weights, num_samples):
    """
    Hierarchical sampling based on coarse network weights
    
    Args:
        ray_origins: [N_rays, 3] ray origins
        ray_directions: [N_rays, 3] ray directions
        z_vals: [N_rays, N_samples_coarse] coarse depth values
        weights: [N_rays, N_samples_coarse] coarse weights
        num_samples: number of fine samples
        
    Returns:
        points: [N_rays, N_samples, 3] sampled points
        z_vals_combined: [N_rays, N_samples] combined depth values
    """
    # Check for invalid weights
    if torch.any(torch.isnan(weights)) or torch.any(torch.isinf(weights)):
        print("Warning: NaN or Inf in weights, using uniform sampling")
        # Fallback to uniform sampling between near and far
        near = z_vals[..., 0:1]
        far = z_vals[..., -1:]
        t = torch.linspace(0., 1., num_samples, device=ray_origins.device)
        z_samples = near + (far - near) * t[None, :]
        z_vals_combined = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)[0]
        points = ray_origins[:, None, :] + ray_directions[:, None, :] * z_vals_combined[..., None]
        return points, z_vals_combined
    
    # Prevent division by zero and clamp weights
    weights = torch.clamp(weights, min=1e-8, max=1e8)
    
    # Normalize weights to get PDF
    weight_sum = torch.sum(weights, dim=-1, keepdim=True)
    weight_sum = torch.clamp(weight_sum, min=1e-8)
    pdf = weights / weight_sum
    
    # Calculate CDF with numerical stability
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
    cdf = torch.clamp(cdf, 0., 1.)  # Ensure CDF is bounded
    
    # Sample from CDF using inverse transform sampling
    u = torch.rand(ray_origins.shape[0], num_samples, device=ray_origins.device)
    u = torch.clamp(u, 1e-6, 1. - 1e-6)  # Avoid boundary values
    u = u.contiguous()
    
    # Find indices in CDF with proper bounds
    indices = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(indices - 1, min=0, max=cdf.shape[-1] - 1)
    above = torch.clamp(indices, min=0, max=cdf.shape[-1] - 1)
    
    # Make sure indices are valid for z_vals
    below_z = torch.clamp(below, min=0, max=z_vals.shape[-1] - 1) 
    above_z = torch.clamp(above, min=0, max=z_vals.shape[-1] - 1)
    
    # Gather CDF values
    cdf_below = torch.gather(cdf, -1, below)
    cdf_above = torch.gather(cdf, -1, above)
    
    # Gather z values
    z_below = torch.gather(z_vals, -1, below_z)
    z_above = torch.gather(z_vals, -1, above_z)
    
    # Linear interpolation with numerical stability
    denom = cdf_above - cdf_below
    denom = torch.clamp(denom, min=1e-8)
    t = torch.clamp((u - cdf_below) / denom, 0., 1.)
    z_samples = z_below + t * (z_above - z_below)
    
    # Clamp z_samples to valid range
    z_min, z_max = z_vals.min(), z_vals.max()
    z_samples = torch.clamp(z_samples, z_min, z_max)
    
    # Combine with coarse samples
    z_vals_combined, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
    
    # Calculate points
    points = ray_origins[:, None, :] + ray_directions[:, None, :] * z_vals_combined[..., None]
    
    return points, z_vals_combined
