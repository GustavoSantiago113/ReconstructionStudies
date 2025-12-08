import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import open3d as o3d
from torch.cuda.amp import GradScaler, autocast

from utils.nerf_model import (
    NeRFModel, 
    volume_rendering, 
    sample_along_rays, 
    hierarchical_sampling
)
from utils.colmap_loader import COLMAPDataset, qvec2rotmat


def get_rays(H, W, K, pose):
    """
    Generate rays from camera parameters
    
    Args:
        H, W: image height and width
        K: 3x3 intrinsics matrix
        pose: 4x4 camera-to-world matrix
        
    Returns:
        rays_o: [H*W, 3] ray origins
        rays_d: [H*W, 3] ray directions
    """
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing='xy'
    )
    
    # Convert to camera coordinates
    dirs = torch.stack([
        (i - K[0, 2]) / K[0, 0],
        -(j - K[1, 2]) / K[1, 1],  # Negative for right-handed coordinate system
        -torch.ones_like(i)
    ], dim=-1)
    
    # Transform to world coordinates
    rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    rays_o = pose[:3, 3].expand(rays_d.shape)
    
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

class RayDataset(Dataset):
    def __init__(self, colmap_dataset, image_ids, device='cuda'):
        self.device = device
        self.rays_o = []
        self.rays_d = []
        self.rgb_gt = []
        
        print(f"Processing {len(image_ids)} images for ray dataset...")
        
        for img_id in image_ids:
            data = colmap_dataset.images[img_id]
            camera_id = data["camera_id"]
            camera = colmap_dataset.cameras[camera_id]
            
            # Get pose matrices
            R = qvec2rotmat(data["qvec"])
            t = data["tvec"]
            
            # Load image - keep on CPU initially, move to GPU in batches
            from PIL import Image
            img_path = Path(colmap_dataset.image_dir) / data["name"]
            img = Image.open(img_path).convert('RGB')
            
            # Resize image to reduce memory usage
            max_size = 800  # Reduce image size to save memory
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size)
                # Update camera parameters for resized image
                camera["width"] = new_size[0]
                camera["height"] = new_size[1]
                camera["params"] = tuple(p * ratio for p in camera["params"][:2]) + camera["params"][2:]
            
            img = torch.from_numpy(np.array(img)).float() / 255.0
            H, W = img.shape[:2]
            
            # Generate rays for this image
            pose = torch.eye(4)
            pose[:3, :3] = torch.from_numpy(R).float()
            pose[:3, 3] = torch.from_numpy(t).float()
            K = torch.zeros((3, 3), dtype=torch.float32)
            K[0, 0] = camera["params"][0]  # fx
            K[1, 1] = camera["params"][1]  # fy
            K[0, 2] = camera["params"][2]  # cx
            K[1, 2] = camera["params"][3]  # cy
            K[2, 2] = 1.0
            rays_o, rays_d = get_rays(H, W, K, pose)
            
            # Store data (keep on CPU to save GPU memory during loading)
            self.rays_o.append(rays_o.cpu())
            self.rays_d.append(rays_d.cpu())
            self.rgb_gt.append(img.reshape(-1, 3).cpu())
        
        # Concatenate all rays
        self.rays_o = torch.cat(self.rays_o, dim=0)
        self.rays_d = torch.cat(self.rays_d, dim=0)
        self.rgb_gt = torch.cat(self.rgb_gt, dim=0)
        
        print(f"Ray dataset created: {len(self.rays_o)} rays")
        
    def __len__(self):
        return len(self.rays_o)
    
    def __getitem__(self, idx):
        # Return tensors directly instead of dictionary
        return self.rays_o[idx], self.rays_d[idx], self.rgb_gt[idx]


def train_nerf(
    colmap_dir,
    output_dir,
    image_dir=None,
    num_epochs=100,
    batch_size=1024,
    num_coarse_samples=64,
    num_fine_samples=128,
    lr=5e-4,
    device='cuda'
):
    """
    Train NeRF model from COLMAP data
    
    Args:
        colmap_dir: path to COLMAP sparse reconstruction
        output_dir: where to save checkpoints and outputs
        image_dir: path to directory containing images (optional, will try to auto-detect)
        num_epochs: number of training epochs
        batch_size: number of rays per batch
        num_coarse_samples: samples for coarse network
        num_fine_samples: samples for fine network
        lr: learning rate
        device: 'cuda' or 'cpu'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load COLMAP data
    print("Loading COLMAP dataset...")
    colmap_path = Path(colmap_dir)
    
    if image_dir is None:
        # Try to find image directory - common locations relative to sparse directory
        possible_image_dirs = [
            colmap_path.parent.parent / "images",  # ../images from sparse/0
            colmap_path.parent / "images",         # images from sparse
            colmap_path / "../../images",          # ../../images from sparse/0
        ]
        
        for img_dir in possible_image_dirs:
            if img_dir.exists() and any(img_dir.iterdir()):
                image_dir = str(img_dir)
                print(f"Found images in: {image_dir}")
                break
        
        if image_dir is None:
            # Fallback: use current directory or ask user to specify
            print("Warning: Could not find image directory automatically")
            print("Please ensure images are available or modify the path")
            image_dir = str(colmap_path.parent.parent / "images")  # Default guess
    
    dataset = COLMAPDataset(colmap_dir, image_dir)
    train_ids, test_ids = dataset.get_train_test_split(test_every=8)
    
    print(f"Train images: {len(train_ids)}, Test images: {len(test_ids)}")
    
    # Create ray dataset
    print("Creating ray dataset...")
    ray_dataset = RayDataset(dataset, train_ids, device=device)
    # Optimize DataLoader for GPU memory usage
    num_workers = min(2, os.cpu_count()) if device == 'cuda' else 0
    dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': True if device == 'cuda' else False,
    }
    
    # Add worker-specific parameters only when using multiple workers
    if num_workers > 0:
        dataloader_kwargs.update({
            'persistent_workers': True,
            'prefetch_factor': 2
        })
    
    dataloader = DataLoader(ray_dataset, **dataloader_kwargs)
    
    # Initialize models
    print("Initializing models...")
    model_coarse = NeRFModel().to(device)
    model_fine = NeRFModel().to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        list(model_coarse.parameters()) + list(model_fine.parameters()),
        lr=lr
    )
    
    # Mixed precision scaler for faster training on GPU
    scaler = GradScaler() if device == 'cuda' else None
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    # Scene bounds for sampling
    near = dataset.bounds_radius * 0.5
    far = dataset.bounds_radius * 2.5
    
    print(f"Sampling bounds: near={near:.3f}, far={far:.3f}")
    
    # Training loop
    print("Starting training...")
    accumulation_steps = 2  # Accumulate gradients over 2 batches
    for epoch in range(num_epochs):
        model_coarse.train()
        model_fine.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (rays_o, rays_d, rgb_gt) in enumerate(pbar):
            rays_o = rays_o.to(device, non_blocking=True)
            rays_d = rays_d.to(device, non_blocking=True)
            rgb_gt = rgb_gt.to(device, non_blocking=True)
            
            # Use mixed precision for forward pass
            with autocast(enabled=(scaler is not None)):
                # Coarse sampling
                points_coarse, z_vals_coarse = sample_along_rays(
                    rays_o, rays_d, near, far, num_coarse_samples, perturb=True
                )
                
                # Flatten for network
                N_rays, N_samples = points_coarse.shape[:2]
                points_flat = points_coarse.reshape(-1, 3)
                dirs_flat = rays_d[:, None, :].expand(N_rays, N_samples, 3).reshape(-1, 3)
                
                # Coarse network forward pass
                rgb_coarse, sigma_coarse = model_coarse(points_flat, dirs_flat)
                rgb_coarse = rgb_coarse.reshape(N_rays, N_samples, 3)
                sigma_coarse = sigma_coarse.reshape(N_rays, N_samples, 1)
                
                # Volume rendering (coarse)
                rgb_map_coarse, depth_map_coarse, acc_map_coarse, weights_coarse = volume_rendering(
                    rgb_coarse, sigma_coarse, z_vals_coarse, rays_d
                )
                
                # Debug: Check for numerical issues before hierarchical sampling
                if torch.any(torch.isnan(weights_coarse)) or torch.any(torch.isinf(weights_coarse)):
                    print(f"Warning: Invalid weights detected at batch {batch_idx}")
                    print(f"Weights stats: min={weights_coarse.min():.6f}, max={weights_coarse.max():.6f}")
                    print(f"NaN count: {torch.sum(torch.isnan(weights_coarse))}")
                    print(f"Inf count: {torch.sum(torch.isinf(weights_coarse))}")
                
                # Hierarchical sampling
                points_fine, z_vals_fine = hierarchical_sampling(
                    rays_o, rays_d, z_vals_coarse, weights_coarse.detach(), num_fine_samples
                )
                
                # Fine network forward pass
                N_samples_fine = points_fine.shape[1]
                points_flat = points_fine.reshape(-1, 3)
                dirs_flat = rays_d[:, None, :].expand(N_rays, N_samples_fine, 3).reshape(-1, 3)
                
                rgb_fine, sigma_fine = model_fine(points_flat, dirs_flat)
                rgb_fine = rgb_fine.reshape(N_rays, N_samples_fine, 3)
                sigma_fine = sigma_fine.reshape(N_rays, N_samples_fine, 1)
                
                # Volume rendering (fine)
                rgb_map_fine, depth_map_fine, acc_map_fine, weights_fine = volume_rendering(
                    rgb_fine, sigma_fine, z_vals_fine, rays_d
                )
                
                # Photometric loss (MSE between rendered and ground truth RGB)
                loss_coarse = torch.mean((rgb_map_coarse - rgb_gt) ** 2)
                loss_fine = torch.mean((rgb_map_fine - rgb_gt) ** 2)
                loss = (loss_coarse + loss_fine) / accumulation_steps
            
            # Backward pass with mixed precision
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            epoch_loss += loss.item() * accumulation_steps
            num_batches += 1
            pbar.set_postfix({'loss': loss.item() * accumulation_steps, 'loss_fine': loss_fine.item()})
            if batch_idx % 100 == 0 and device == 'cuda':
                torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.6f}")
        
        # Step scheduler
        scheduler.step()
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_coarse_state_dict': model_coarse.state_dict(),
                'model_fine_state_dict': model_fine.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = output_dir / "nerf_final.pth"
    torch.save({
        'model_coarse_state_dict': model_coarse.state_dict(),
        'model_fine_state_dict': model_fine.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    print(f"Saved final model to {final_path}")
    
    return model_coarse, model_fine, dataset


def extract_point_cloud(
    model,
    dataset,
    output_path,
    resolution=128,
    density_threshold=10.0,
    device='cuda'
):
    """
    Extract point cloud from trained NeRF model
    
    Args:
        model: trained NeRF model (use fine model)
        dataset: COLMAPDataset instance
        output_path: where to save point cloud
        resolution: grid resolution for sampling
        density_threshold: minimum density for a point to be included
        device: 'cuda' or 'cpu'
    """
    print(f"Extracting point cloud with resolution {resolution}...")
    model.eval()
    
    # Create 3D grid within scene bounds
    x = torch.linspace(dataset.bounds_min[0], dataset.bounds_max[0], resolution)
    y = torch.linspace(dataset.bounds_min[1], dataset.bounds_max[1], resolution)
    z = torch.linspace(dataset.bounds_min[2], dataset.bounds_max[2], resolution)
    
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3).to(device)
    
    # Query density at grid points
    batch_size = 10000
    all_densities = []
    all_colors = []
    
    # Use a fixed view direction (or average of all camera directions)
    view_dir = torch.tensor([0.0, 0.0, -1.0], device=device).expand(batch_size, 3)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(points), batch_size), desc="Querying NeRF"):
            batch_points = points[i:i+batch_size]
            batch_view = view_dir[:len(batch_points)]
            
            rgb, sigma = model(batch_points, batch_view)
            all_densities.append(sigma.squeeze().cpu().numpy())
            all_colors.append(rgb.cpu().numpy())
    
    densities = np.concatenate(all_densities)
    colors = np.concatenate(all_colors)
    points_np = points.cpu().numpy()
    
    # Filter by density threshold
    mask = densities > density_threshold
    filtered_points = points_np[mask]
    filtered_colors = colors[mask]
    
    print(f"Extracted {len(filtered_points)} points (from {len(points)} sampled)")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    # Save point cloud
    output_path = Path(output_path)
    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"Saved point cloud to {output_path}")
    
    return pcd


if __name__ == "__main__":
    # Example usage
    colmap_dir = "path/to/colmap/sparse/0"
    output_dir = "nerf_output"
    
    # Train NeRF
    model_coarse, model_fine, dataset = train_nerf(
        colmap_dir=colmap_dir,
        output_dir=output_dir,
        num_epochs=50,
        batch_size=1024,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Extract point cloud
    pcd = extract_point_cloud(
        model=model_fine,
        dataset=dataset,
        output_path=os.path.join(output_dir, "point_cloud.ply"),
        resolution=128,
        density_threshold=10.0
    )
    
    # Visualize
    o3d.visualization.draw_geometries([pcd])
