import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from typing import Dict, Tuple, Optional
from PIL import Image
import os


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


class SimpleNeRFTrainer:
    """Trainer for simplified NeRF"""
    
    def __init__(self, data_dir: str, output_dir: str, device: str = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load data
        self.load_data()
        
        # Initialize model
        self.model = SimpleNeRF().to(self.device)
        
        # Scene normalization (world -> [-1, 1])
        cam_centers = self.poses[:, :3, 3]
        self.scene_center = cam_centers.mean(dim=0)
        distances = (cam_centers - self.scene_center).norm(dim=1)
        self.scene_radius = distances.max() * 2.0  # Give more room
        self.scene_radius = torch.clamp(self.scene_radius, min=1.0)
        print(f"Scene center: {self.scene_center.cpu().numpy()}, radius: {float(self.scene_radius):.3f}")
        print(f"Camera distance range: {distances.min():.3f} to {distances.max():.3f}")
    
    def normalize_positions(self, pts: torch.Tensor) -> torch.Tensor:
        """Map world-space positions to [-1, 1] using scene_center/radius."""
        return torch.clamp((pts - self.scene_center) / self.scene_radius, -1.0, 1.0)
        
    def load_data(self):
        """Load images and camera poses"""
        transforms_file = self.data_dir / "transforms.json"
        
        with open(transforms_file, 'r') as f:
            transforms = json.load(f)
        
        self.images = []
        self.poses = []
        print("Loading images and poses...")
        for frame in transforms['frames']:
            img_path = self.data_dir / frame['file_path'].lstrip('./')
            
            if img_path.exists():
                img = Image.open(img_path)
                img = img.convert('RGB')
                img = np.array(img) / 255.0
                self.images.append(img)
                
                pose = np.array(frame['transform_matrix'])
                self.poses.append(pose)
        
        self.images = torch.FloatTensor(np.stack(self.images)).to(self.device)
        self.poses = torch.FloatTensor(np.stack(self.poses)).to(self.device)
        
        # Use actual image size from loaded data (already resized during preprocessing)
        self.img_h, self.img_w = self.images.shape[1], self.images.shape[2]
        
        # Camera intrinsics (should already be scaled for the resized images)
        self.fx = float(transforms.get('fl_x', transforms.get('fx', max(self.img_w, 1))))
        self.fy = float(transforms.get('fl_y', transforms.get('fy', self.fx)))
        self.cx = float(transforms.get('cx', self.img_w / 2))
        self.cy = float(transforms.get('cy', self.img_h / 2))
        
        print(f"Loaded {len(self.images)} images")
        print(f"Image size: {self.img_h}x{self.img_w}")
        print(f"Intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")
    
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
        rays_o = pose[:3, 3].expand(rays_d.shape)
        
        return rays_o, rays_d
    
    def render_rays(self, rays_o, rays_d, near=0.2, far=8.0, n_samples=64):
        """Volume rendering along rays"""
        # Sample points along rays
        t_vals = torch.linspace(0, 1, n_samples, device=self.device)
        z_vals = near * (1 - t_vals) + far * t_vals
        z_vals = z_vals.expand(rays_o.shape[0], n_samples)
        
        # Get sample points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        pts_flat = pts.reshape(-1, 3)
        # Normalize to hash grid domain [-1, 1]
        pts_flat_norm = self.normalize_positions(pts_flat)
        
        # Get directions
        dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        dirs_flat = dirs[:, None, :].expand_as(pts).reshape(-1, 3)
        
        # Query network
        colors, densities = self.model(pts_flat_norm, dirs_flat)
        
        colors = colors.reshape(*pts.shape[:-1], 3)
        densities = densities.reshape(*pts.shape[:-1])
        
        # Volume rendering
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e10], dim=-1)
        
        # Densities are already processed by softplus in the model
        sigma = densities.squeeze(-1)
        alpha = 1.0 - torch.exp(-sigma * dists)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
            dim=-1
        )[..., :-1]
        
        rgb = torch.sum(weights[..., None] * colors, dim=-2)
        
        return rgb, weights, z_vals

    def render_image(self, img_idx: int = 0, near: float = 0.2, far: float = 8.0, n_samples: int = 64, chunk: int = 8192):
        pose = self.poses[img_idx]
        H, W = self.img_h, self.img_w
        rays_o, rays_d = self.get_rays(pose, H, W)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        all_rgb = []
        all_depth = []
        with torch.no_grad():
            for i in range(0, rays_o.shape[0], chunk):
                ro = rays_o[i:i+chunk]
                rd = rays_d[i:i+chunk]
                rgb, weights, z_vals = self.render_rays(ro, rd, near=near, far=far, n_samples=n_samples)
                if weights is not None and z_vals is not None:
                    depth = (weights * z_vals).sum(dim=-1)
                else:
                    depth = torch.zeros(rgb.shape[0], device=rgb.device)
                all_rgb.append(rgb)
                all_depth.append(depth)
        rgb = torch.cat(all_rgb, dim=0).reshape(H, W, 3).clamp(0, 1).cpu().numpy()
        depth = torch.cat(all_depth, dim=0).reshape(H, W).cpu().numpy()
        return rgb, depth

    def save_preview(self, img_idx: int = 0, near: float = 0.2, far: float = 8.0, n_samples: int = 64, prefix: str = "preview"):
        rgb, depth = self.render_image(img_idx, near=near, far=far, n_samples=n_samples)
        
        # Debug prints
        print(f"  RGB range: [{rgb.min():.4f}, {rgb.max():.4f}]")
        print(f"  Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
        
        rgb_img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        
        # Normalize depth for visualization
        d = depth.copy()
        if d.max() > 0:
            d_norm = d / d.max()  # Normalize to [0,1]
        else:
            d_norm = np.zeros_like(d)
        d_img = (d_norm * 255).astype(np.uint8)
        
        Image.fromarray(rgb_img).save(self.output_dir / f"{prefix}_rgb_{img_idx:03d}.png")
        Image.fromarray(d_img).save(self.output_dir / f"{prefix}_depth_{img_idx:03d}.png")
        print(f"  Saved preview: {prefix}_rgb_{img_idx:03d}.png, {prefix}_depth_{img_idx:03d}.png")
    
    def train(self, n_iters: int = 10000, batch_size: int = 1024, lr: float = 5e-4, preview_every: int = 0, preview_index: int = 0):
        """Train the NeRF model, saving only the best model (lowest loss)."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        print(f"\nTraining for {n_iters} iterations...")
        best_loss = float('inf')
        best_state = None
        best_iter = 0
        for i in range(n_iters):
            # Random image
            img_idx = np.random.randint(0, len(self.images))
            img = self.images[img_idx]  # Already on device
            pose = self.poses[img_idx]  # Already on device
            # Get rays
            rays_o, rays_d = self.get_rays(pose, self.img_h, self.img_w)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            # Random ray batch
            select_idx = np.random.choice(rays_o.shape[0], batch_size, replace=False)
            rays_o_batch = rays_o[select_idx]
            rays_d_batch = rays_d[select_idx]
            target_rgb = img.reshape(-1, 3)[select_idx]
            # Render
            rgb, weights, z_vals = self.render_rays(rays_o_batch, rays_d_batch)
            # Compute loss
            loss = F.mse_loss(rgb, target_rgb)
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Log
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{n_iters}, Loss: {loss.item():.6f}")
            if preview_every > 0 and (i + 1) % preview_every == 0:
                print("Rendering preview (RGB + depth)...")
                self.debug_network_output()  # Debug network before rendering
                self.save_preview(preview_index, prefix=f"iter_{i+1}")
            # Track best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = self.model.state_dict()
                best_iter = i + 1
        print(f"\n\u2713 Training completed! Best loss: {best_loss:.6f} at iteration {best_iter}")
        # Save only the best checkpoint
        self.save_best_checkpoint(best_state, best_iter, best_loss)

    def save_best_checkpoint(self, state_dict, iteration, loss):
        """Save only the best model checkpoint."""
        checkpoint_path = self.output_dir / f"nerf_best.pth"
        torch.save({
            'iteration': iteration,
            'loss': loss,
            'model_state_dict': state_dict,
        }, checkpoint_path)
        print(f"  Saved best checkpoint: {checkpoint_path} (iter {iteration}, loss {loss:.6f})")
    
    def save_checkpoint(self, iteration):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / f"nerf_{iteration}.pth"
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
        }, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")
    
    def extract_mesh(self, resolution: int = 128, threshold: Optional[float] = None):
        """Extract mesh using marching cubes over opacity in normalized space."""
        try:
            from skimage import measure
            import trimesh
        except ImportError:
            print("\u2717 Please install: pip install scikit-image trimesh")
            return None
        
        print(f"\nExtracting mesh (resolution: {resolution})...")
        
        # Create grid in normalized coordinates [-1, 1]
        bound = 1.0
        x = torch.linspace(-bound, bound, resolution, device=self.device)
        y = torch.linspace(-bound, bound, resolution, device=self.device)
        z = torch.linspace(-bound, bound, resolution, device=self.device)
        
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        positions = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        
        # Dummy directions (not used for density)
        directions = torch.zeros_like(positions)
        
        # Query opacity in batches (convert sigma -> alpha using grid step)
        batch_size = 10000
        alphas = []
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(positions), batch_size):
                batch_pos = positions[i:i+batch_size]
                batch_dir = directions[i:i+batch_size]
                # Positions already in normalized space
                _, sigma = self.model(batch_pos, batch_dir)
                # Step size in world coordinates for proper opacity calculation
                dt = (2.0 * float(self.scene_radius)) / resolution
                # Sigma is already activated by softplus in the model
                alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dt)
                alphas.append(alpha.cpu())
        
        alpha = torch.cat(alphas, dim=0)
        alpha_grid = alpha.reshape(resolution, resolution, resolution).numpy()
        
        # Opacity diagnostics
        print(f"  Opacity range: [{alpha_grid.min():.4f}, {alpha_grid.max():.4f}]")
        print(f"  Opacity mean: {alpha_grid.mean():.4f}, std: {alpha_grid.std():.4f}")
        
        # Adaptive threshold if not provided (use 50th percentile of non-zero values)
        if threshold is None:
            non_zero = alpha_grid[alpha_grid > 1e-6]
            if len(non_zero) > 0:
                threshold = float(np.percentile(non_zero, 50))
            else:
                threshold = 1e-4
        threshold = float(np.clip(threshold, 1e-6, 0.9))
        print(f"  Using opacity threshold: {threshold:.6f}")
        
        # Marching cubes
        print("  Running marching cubes...")
        try:
            verts, faces, normals, values = measure.marching_cubes(
                alpha_grid,
                level=threshold,
                spacing=(2*bound/resolution, 2*bound/resolution, 2*bound/resolution)
            )
            
            # Convert from normalized to world coordinates
            verts = (verts - bound) * float(self.scene_radius.cpu()) + self.scene_center.cpu().numpy()
            
            # Create mesh
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
            
            # Save mesh
            mesh_path = self.output_dir / "mesh.ply"
            mesh.export(mesh_path)
            print(f"\u2713 Mesh saved: {mesh_path}")
            
            return mesh
            
        except Exception as e:
            print(f"\u2717 Marching cubes failed: {e}")
            return None
    
    def extract_pointcloud(self, num_points: int = 100000):
        """Extract point cloud by sampling from density field"""
        try:
            import trimesh
        except ImportError:
            print("\u2717 Please install: pip install trimesh")
            return None
        
        print(f"\\nExtracting point cloud ({num_points} points)...")
        
        # Sample random points in normalized space
        bound = 1.0
        positions_norm = torch.FloatTensor(num_points, 3).uniform_(-bound, bound).to(self.device)
        directions = torch.zeros_like(positions_norm)
        
        # Query colors and alpha
        self.model.eval()
        with torch.no_grad():
            colors, sigma = self.model(positions_norm, directions)
        # Use proper step size in world coordinates
        dt = (2.0 * float(self.scene_radius)) / 128
        alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dt)
        alpha_np = alpha.cpu().numpy()
        print(f"  Alpha range: [{alpha_np.min():.4f}, {alpha_np.max():.4f}]")
        print(f"  Alpha mean: {alpha_np.mean():.4f}, std: {alpha_np.std():.4f}")
        
        # Adaptive alpha threshold (median of non-zero values)
        non_zero = alpha_np[alpha_np > 1e-6]
        if len(non_zero) > 0:
            threshold = float(np.percentile(non_zero, 50))
        else:
            threshold = 1e-6
        print(f"  Using adaptive alpha threshold: {threshold:.6f}")
        
        mask = alpha > threshold
        
        # Un-normalize to world coordinates
        points = (positions_norm[mask] * self.scene_radius + self.scene_center).cpu().numpy()
        colors = (colors[mask].cpu().numpy() * 255).astype(np.uint8)
        
        print(f"  Filtered to {len(points)} points above threshold")
        
        # Create point cloud
        pc = trimesh.PointCloud(points, colors=colors)
        
        # Save
        pc_path = self.output_dir / "pointcloud.ply"
        pc.export(pc_path)
        print(f"\u2713 Point cloud saved: {pc_path}")
        
        return pc


def train_simple_nerf(data_dir: str, output_dir: str, 
                     n_iters: int = 10000,
                     mesh_resolution: int = 128,
                     num_points: int = 100000,
                     preview_every: int = 0,
                     preview_index: int = 0):
    """
    Convenience function to train Instant NGP and extract outputs.
    
    Args:
        data_dir: Directory with transforms.json and images
        output_dir: Output directory
        n_iters: Training iterations
        mesh_resolution: Marching cubes resolution
        num_points: Point cloud size
    """
    print("="*70)
    print("INSTANT NGP TRAINING (Hash Encoding)")
    print("="*70)
    
    # Train
    trainer = SimpleNeRFTrainer(data_dir, output_dir)
    trainer.train(n_iters=n_iters, preview_every=preview_every, preview_index=preview_index)
    
    # Extract mesh
    trainer.extract_mesh(resolution=mesh_resolution)
    
    # Extract point cloud
    trainer.extract_pointcloud(num_points=num_points)
    
    print("="*70)
    print("\u2713 Instant NGP reconstruction completed!")
    print("="*70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simple NeRF training and preview rendering")
    parser.add_argument("--data_dir", type=str, default="../reconstructions/InstantNGP_preprocessed/2_segmentation")
    parser.add_argument("--output_dir", type=str, default="../reconstructions/InstantNGP_preprocessed/3_reconstruction")
    parser.add_argument("--n_iters", type=int, default=10000)
    parser.add_argument("--mesh_resolution", type=int, default=128)
    parser.add_argument("--num_points", type=int, default=100000)
    parser.add_argument("--preview_every", type=int, default=0, help="Render RGB/depth previews every N iters (0=disable)")
    parser.add_argument("--preview_index", type=int, default=0, help="Image index to use for preview rendering")
    parser.add_argument("--render_preview", action="store_true", help="Render a preview without training")
    parser.add_argument("--near", type=float, default=0.5)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()

    if args.render_preview:
        trainer = SimpleNeRFTrainer(args.data_dir, args.output_dir)
        print("Rendering preview (RGB + depth)...")
        trainer.save_preview(args.preview_index, near=args.near, far=args.far, n_samples=args.samples)
    else:
        train_simple_nerf(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_iters=args.n_iters,
            mesh_resolution=args.mesh_resolution,
            num_points=args.num_points,
            preview_every=args.preview_every,
            preview_index=args.preview_index,
        )
