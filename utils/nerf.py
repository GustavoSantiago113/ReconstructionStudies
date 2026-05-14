import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.init as init
import time
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import math
import skimage.measure as measure
import trimesh

def encoding(x, L=10):
  res = [x]
  for i in range(L):
    for fn in [torch.sin, torch.cos]:
      res.append(fn(2 ** i * torch.pi * x))
  return torch.cat(res,dim=-1)

class NeRF(nn.Module):
  def __init__(self, pos_enc_dim=63, view_enc_dim=27, hidden=256) -> None:
     super().__init__()

     self.linear1 = nn.Sequential(nn.Linear(pos_enc_dim,hidden),nn.ReLU())

     self.pre_skip_linear = nn.Sequential()
     for _ in range(4):
      self.pre_skip_linear.append(nn.Linear(hidden,hidden))
      self.pre_skip_linear.append(nn.ReLU())

     self.linear_skip = nn.Sequential(nn.Linear(pos_enc_dim+hidden,hidden),nn.ReLU())

     self.post_skip_linear = nn.Sequential()
     for _ in range(2):
      self.post_skip_linear.append(nn.Linear(hidden,hidden))
      self.post_skip_linear.append(nn.ReLU())

     self.density_layer = nn.Sequential(nn.Linear(hidden,1),nn.ReLU())

     self.linear2 = nn.Linear(hidden,hidden)

     self.color_linear1 = nn.Sequential(nn.Linear(hidden+view_enc_dim,hidden//2),nn.ReLU())
     self.color_linear2 = nn.Sequential(nn.Linear(hidden//2,3),nn.Sigmoid())

     self._init_weights()

  def _init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)
    # Use xavier for the sigmoid output layer
    for m in self.color_linear2.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)

  def forward(self, input):

    positions = input[...,:3]
    view_dirs = input[...,3:]

    # Encode
    pos_enc = encoding(positions,L=10)
    view_enc = encoding(view_dirs,L=4)

    x = self.linear1(pos_enc)
    x = self.pre_skip_linear(x)

    # Skip connection
    x = torch.cat([x,pos_enc],dim=-1)
    x = self.linear_skip(x)

    x = self.post_skip_linear(x)

    # Density
    sigma = self.density_layer(x)

    x = self.linear2(x)

    # View Encoding
    x = torch.cat([x,view_enc],dim=-1)
    x = self.color_linear1(x)

    # Color Prediction
    rgb = self.color_linear2(x)

    return torch.cat([sigma,rgb],dim=-1)

def get_rays(H, W, focal, c2w):
  """
  Generate rays for a given camera configuration.

  Args:
    H: Image height.
    W: Image width.
    focal: Focal length.
    c2w: Camera-to-world transformation matrix (4x4).

  Returns:
    rays_o: Ray origins (H*W, 3).
    rays_d: Ray directions (H*W, 3).
  """
  device = c2w.device  # Get the device of c2w
  focal = torch.from_numpy(focal).to(device)
  # print(type(H), type(W), type(focal), type(c2w))

  i, j = torch.meshgrid(
      torch.arange(W, dtype=torch.float32, device=device),
      torch.arange(H, dtype=torch.float32, device=device),
      indexing='xy'
  )
  dirs = torch.stack(
      [(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i, device = device)], -1
  )

  rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
  rays_d = rays_d.view(-1, 3)
  rays_o = c2w[:3, -1].expand(rays_d.shape)

  return rays_o, rays_d

def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, device, rand=False, embed_fn=None, chunk=1024*4):
    def batchify(fn, chunk):
        return lambda inputs: torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    # Sampling
    z_vals = torch.linspace(near, far, steps=N_samples, device=device)

    z_vals = z_vals.unsqueeze(0).expand(rays_o.shape[0], -1).clone()
    if rand:
        z_vals += torch.rand(rays_o.shape[0], N_samples, device=device) * (far - near) / N_samples

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

    # Normalize view directions
    view_dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    view_dirs = view_dirs[..., None, :].expand(pts.shape)

    input_pts = torch.cat((pts, view_dirs), dim=-1)
    raw = batchify(network_fn, chunk)(input_pts)

    # Apply activations here instead of in network
    sigma_a = raw[...,0]  # Shape: [batch, N_samples]
    rgb = raw[...,1:]    # Shape: [batch, N_samples, 3]

    # Improved volume rendering
    dists = z_vals[..., 1:] - z_vals[..., :-1]  # Shape: [batch, N_samples-1]
    dists = torch.cat([dists, torch.full((dists.shape[0], 1), 1e10, device=device)], -1)

    # No need to manually expand dists as broadcasting will handle it
    alpha = 1. - torch.exp(-sigma_a * dists)  # Shape: [batch, N_samples]
    alpha = alpha.unsqueeze(-1)  # Shape: [batch, N_samples, 1]

    # Computing transmittance
    ones_shape = (alpha.shape[0], 1, 1)
    T = torch.cumprod(
        torch.cat([
            torch.ones(ones_shape, device=device),
            1. - alpha + 1e-10
        ], dim=1),
        dim=1
    )[:, :-1]  # Shape: [batch, N_samples, 1]

    weights = alpha * T  # Shape: [batch, N_samples, 1]

    # Compute final colors and depths
    rgb_map = torch.sum(weights * rgb, dim=1)  # Sum along sample dimension
    depth_map = torch.sum(weights.squeeze(-1) * z_vals, dim=-1)  # Shape: [batch]
    acc_map = torch.sum(weights.squeeze(-1), dim=-1)  # Shape: [batch]

    return rgb_map, depth_map, acc_map

def train(images, poses, H, W, focal, testpose, testimg, n_iter, n_samples, i_plot, i_val, device, batch_size=4096):
    use_amp = (device == "cuda")
    print(f"Using device: {device}")
    model = NeRF().to(device)

    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    warmup_iters = n_iter // 20
    def lr_fn(step):
        if step < warmup_iters:
            return max(0.01, step / max(1, warmup_iters))
        progress = (step - warmup_iters) / max(1, n_iter - warmup_iters)
        return max(1e-5 / 5e-4, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    amp_device = "cuda" if device == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)

    psnrs = []
    iternums = []
    t = time.time()

    images_tensor = torch.from_numpy(images).float().to(device)
    poses_tensor = torch.from_numpy(poses).float().to(device)

    # Pre-compute all rays for stable random-ray sampling
    all_rays_o = []
    all_rays_d = []
    all_rgbs = []
    for idx in range(images.shape[0]):
        pose = poses_tensor[idx]
        rays_o, rays_d = get_rays(H, W, focal, pose)
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)
        all_rgbs.append(images_tensor[idx].reshape(-1, 3))
    all_rays_o = torch.cat(all_rays_o, 0)
    all_rays_d = torch.cat(all_rays_d, 0)
    all_rgbs = torch.cat(all_rgbs, 0)
    n_total_rays = all_rays_o.shape[0]

    pbar = tqdm(range(n_iter), desc="Training NeRF", unit="iter")
    for i in pbar:
        ray_idx = torch.randint(0, n_total_rays, (batch_size,))
        batch_rays_o = all_rays_o[ray_idx]
        batch_rays_d = all_rays_d[ray_idx]
        target = all_rgbs[ray_idx]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            rgb, depth, acc = render_rays(
                model, batch_rays_o, batch_rays_d,
                near=2., far=6., N_samples=n_samples, device=device, rand=True
            )
            loss = criterion(rgb, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if i % i_val == 0:
            with torch.no_grad():
                test_rays_o, test_rays_d = get_rays(H, W, focal, testpose)
                with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                    val_rgb, val_depth, val_acc = render_rays(
                        model, test_rays_o, test_rays_d,
                        near=2., far=6., N_samples=n_samples, device=device
                    )
                    val_rgb = val_rgb.reshape(H, W, 3)
                    val_loss = criterion(val_rgb, testimg)
                    psnr = -10. * torch.log10(val_loss)

            psnrs.append(psnr.item())
            iternums.append(i)
            pbar.set_postfix(loss=f"{loss.item():.6f}", psnr=f"{psnr.item():.2f}")

        if i % i_plot == 0:
            with torch.no_grad():
                test_rays_o, test_rays_d = get_rays(H, W, focal, testpose)
                with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                    plot_rgb, plot_depth, _ = render_rays(
                        model, test_rays_o, test_rays_d,
                        near=2., far=6., N_samples=n_samples, device=device
                    )
                    plot_rgb = plot_rgb.reshape(H, W, 3)

            print(f'Iteration: {i}, Loss: {loss.item():.6f}, Time: {(time.time() - t) / i_plot:.2f} secs per iter')
            t = time.time()

            plt.figure(figsize=(10, 4))
            plt.subplot(141)
            plt.imshow(testimg.cpu().detach())
            plt.title('Ground Truth')
            plt.subplot(142)
            plt.imshow(plot_rgb.cpu().detach())
            plt.title(f'Iteration: {i}')
            plt.subplot(143)
            plt.plot(iternums, psnrs)
            plt.title('PSNR')
            plt.subplot(144)
            depth_img = plot_depth.cpu().detach().reshape(H, W)
            plt.imshow(depth_img, cmap='viridis')
            plt.colorbar()
            plt.title('Depth')
            plt.show()

    return model

def encoding(x: torch.Tensor, L: int) -> torch.Tensor:
    """Positional encoding — must match the one used during training."""
    freqs = 2.0 ** torch.arange(L, dtype=x.dtype, device=x.device)  # (L,)
    x_freq = x[..., None] * freqs                                    # (..., 3, L)
    x_freq = x_freq.reshape(*x.shape[:-1], -1)                       # (..., 3*L)
    return torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=-1) # (..., 6*L)

def render_fn(pts: torch.Tensor, model, device="cuda") -> tuple:
    view_dirs = torch.zeros_like(pts)
    model_input = torch.cat([pts, view_dirs], dim=-1)   # (N, 6)
    with torch.no_grad():
        raw = model(model_input)                         # (N, 4)
    return raw[..., 1:], raw[..., 0]                    # rgb (N,3), sigma (N,)

def extract_mesh(
    model,
    bound: float = 4.0,
    resolution: int = 256,
    density_threshold: float = 10.0,
    chunk: int = 32768,
    device: str = "cuda",
) -> tuple:
    model.eval()
    lin = torch.linspace(-bound, bound, resolution, device=device)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    sigmas = []
    for i in range(0, len(pts), chunk):
        _, sigma = render_fn(pts[i : i + chunk], model, device)
        sigmas.append(sigma.cpu())

    sigma_vol = torch.cat(sigmas).numpy().reshape(resolution, resolution, resolution)

    print(f"Density  min={sigma_vol.min():.3f}  max={sigma_vol.max():.3f}  "
          f"mean={sigma_vol.mean():.3f}  — adjust density_threshold accordingly")

    verts, faces, _, _ = measure.marching_cubes(sigma_vol, level=density_threshold)

    # voxel coords → world coords
    verts = verts / (resolution - 1) * (2 * bound) - bound
    return verts, faces


def color_mesh(
    verts: np.ndarray,
    train_poses: np.ndarray,
    model,
    chunk: int = 4096,
    device: str = "cuda",
) -> np.ndarray:
    """
    For each vertex, query the NeRF model from every training camera direction
    and average the resulting RGB values.
    """
    model.eval()
    verts_t = torch.tensor(verts, dtype=torch.float32, device=device)  # (V, 3)
    
    color_accum  = np.zeros((len(verts), 3), np.float64)
    weight_accum = np.zeros(len(verts),      np.float64)

    for c2w in train_poses:
        cam_pos = c2w[:3, 3]                                    # (3,) world-space camera center
        cam_pos_t = torch.tensor(cam_pos, dtype=torch.float32, device=device)

        # Unit view direction: from vertex toward camera
        dirs = cam_pos_t.unsqueeze(0) - verts_t                 # (V, 3)
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)  # (V, 3) normalized

        rgb_accum_view = []
        sig_accum_view = []

        for i in range(0, len(verts), chunk):
            pts_chunk  = verts_t[i : i + chunk]     # (C, 3)
            dirs_chunk = dirs[i : i + chunk]         # (C, 3)

            model_input = torch.cat([pts_chunk, dirs_chunk], dim=-1)  # (C, 6)
            with torch.no_grad():
                raw = model(model_input)             # (C, 4): [sigma, r, g, b]

            sig_accum_view.append(raw[..., 0].cpu())
            rgb_accum_view.append(raw[..., 1:].cpu())

        sigmas = torch.cat(sig_accum_view).numpy()  # (V,)
        rgbs   = torch.cat(rgb_accum_view).numpy()  # (V, 3)

        # Weight by sigma (confident surface points get more influence)
        # and by camera distance (closer = more reliable)
        dist = np.linalg.norm(verts - cam_pos, axis=1)
        w    = sigmas / (dist + 1e-4)

        color_accum  += rgbs * w[:, None]
        weight_accum += w

    colors = np.full((len(verts), 3), 0.5, np.float32)
    seen   = weight_accum > 0
    colors[seen] = (color_accum[seen] / weight_accum[seen, None]).astype(np.float32)
    return colors


def save_mesh(verts, faces, colors, path="nerf_mesh.glb"):
    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_colors=(colors * 255).clip(0, 255).astype(np.uint8),
        process=False,
    )
    mesh.export(path)
    print(f"Saved → {path}  ({len(verts):,} verts, {len(faces):,} faces)")
    return mesh

def extract_point_cloud(
    model,
    bound: float = 4.0,
    resolution: int = 256,
    density_threshold: float = 10.0,
    chunk: int = 32768,
    device: str = "cuda",
) -> tuple:
    """
    Returns xyz (N, 3) and raw sigmas (N,) for all voxels above threshold.
    """
    model.eval()
    lin = torch.linspace(-bound, bound, resolution, device=device)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)  # (R^3, 3)

    sigmas = []
    for i in range(0, len(pts), chunk):
        view_dirs   = torch.zeros_like(pts[i : i + chunk])
        model_input = torch.cat([pts[i : i + chunk], view_dirs], dim=-1)
        with torch.no_grad():
            raw = model(model_input)
        sigmas.append(raw[..., 0].cpu())

    sigmas = torch.cat(sigmas).numpy()           # (R^3,)
    pts_np = pts.cpu().numpy()                   # (R^3, 3)

    mask   = sigmas > density_threshold
    print(f"Points above threshold: {mask.sum():,} / {len(mask):,}")

    return pts_np[mask], sigmas[mask]

def color_point_cloud(
    xyz: np.ndarray,             # (N, 3)
    sigmas: np.ndarray,          # (N,)
    train_poses: np.ndarray,     # (N_cams, 4, 4)
    model,
    chunk: int = 4096,
    black_threshold: float = 0.15,   # drop points darker than this (0–1)
    device: str = "cuda",
) -> tuple:
    """
    Color each point by querying the NeRF from its nearest training camera.
    Returns filtered (xyz, colors) with black points removed.
    """
    model.eval()
    xyz_t = torch.tensor(xyz, dtype=torch.float32, device=device)

    cam_centers = train_poses[:, :3, 3]                    # (N_cams, 3)

    color_accum  = np.zeros((len(xyz), 3), np.float64)
    weight_accum = np.zeros(len(xyz),      np.float64)

    for c2w in train_poses:
        cam_pos   = c2w[:3, 3]
        cam_pos_t = torch.tensor(cam_pos, dtype=torch.float32, device=device)

        dirs   = cam_pos_t.unsqueeze(0) - xyz_t
        dirs   = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)

        rgbs_view = []
        for i in range(0, len(xyz), chunk):
            model_input = torch.cat([xyz_t[i:i+chunk], dirs[i:i+chunk]], dim=-1)
            with torch.no_grad():
                raw = model(model_input)
            rgbs_view.append(raw[..., 1:].cpu())

        rgbs = torch.cat(rgbs_view).numpy()   # (N, 3) — already sigmoid'd

        dist = np.linalg.norm(xyz - cam_pos, axis=1)
        w    = (sigmas ** 2) / (dist + 1e-4)

        color_accum  += rgbs * w[:, None]
        weight_accum += w

    colors = np.full((len(xyz), 3), 0.5, np.float32)
    seen   = weight_accum > 0
    colors[seen] = (color_accum[seen] / weight_accum[seen, None]).astype(np.float32)

    # ── Remove black points ───────────────────────────────────────────────────
    # A point is "black" if its perceived brightness is below the threshold.
    # Using perceived luminance weights (matches human vision better than mean).
    luminance = (0.2126 * colors[:, 0] +
                 0.7152 * colors[:, 1] +
                 0.0722 * colors[:, 2])

    bright_mask = luminance > black_threshold
    removed = (~bright_mask).sum()
    print(f"Removed {removed:,} black points ({100*removed/len(xyz):.1f}%)")

    return xyz[bright_mask], colors[bright_mask]

def save_point_cloud(
    xyz: np.ndarray,
    colors: np.ndarray,
    path: str = "nerf_pointcloud.ply",
):
    cloud = trimesh.PointCloud(
        vertices=xyz,
        colors=(colors * 255).clip(0, 255).astype(np.uint8),
    )
    cloud.export(path)
    print(f"Saved → {path}  ({len(xyz):,} points)")
    return cloud