import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.init as init
import time
import numpy as np
import os
from tqdm import tqdm
import torch.nn.functional as F
import math

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