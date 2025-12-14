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
        density = F.relu(features[..., :1])
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
        
        # Camera intrinsics
        self.focal = transforms.get('fl_x', 500.0)
        self.cx = transforms.get('cx', transforms.get('w', 512) / 2)
        self.cy = transforms.get('cy', transforms.get('h', 512) / 2)
        self.img_h = transforms.get('h', 512)
        self.img_w = transforms.get('w', 512)
        
        print(f"Loaded {len(self.images)} images")
        print(f"Image size: {self.img_h}x{self.img_w}")
    
    def get_rays(self, pose, H, W, focal):
        """Generate camera rays"""
        i, j = torch.meshgrid(
            torch.arange(W, dtype=torch.float32, device=self.device),
            torch.arange(H, dtype=torch.float32, device=self.device),
            indexing='xy'
        )
        
        dirs = torch.stack([
            (i - W/2) / focal,
            -(j - H/2) / focal,
            -torch.ones_like(i)
        ], dim=-1)
        
        # Rotate ray directions from camera frame to world frame
        rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)
        
        # Origin is the camera position
        rays_o = pose[:3, 3].expand(rays_d.shape)
        
        return rays_o, rays_d
    
    def render_rays(self, rays_o, rays_d, near=0.5, far=6.0, n_samples=64):
        """Volume rendering along rays"""
        # Sample points along rays
        t_vals = torch.linspace(0, 1, n_samples, device=self.device)
        z_vals = near * (1 - t_vals) + far * t_vals
        z_vals = z_vals.expand(rays_o.shape[0], n_samples)
        
        # Get sample points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        pts_flat = pts.reshape(-1, 3)
        
        # Get directions
        dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        dirs_flat = dirs[:, None, :].expand_as(pts).reshape(-1, 3)
        
        # Query network
        colors, densities = self.model(pts_flat, dirs_flat)
        
        colors = colors.reshape(*pts.shape[:-1], 3)
        densities = densities.reshape(*pts.shape[:-1])
        
        # Volume rendering
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e10], dim=-1)
        
        alpha = 1.0 - torch.exp(-F.relu(densities) * dists)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
            dim=-1
        )[..., :-1]
        
        rgb = torch.sum(weights[..., None] * colors, dim=-2)
        
        return rgb, weights, z_vals
    
    def train(self, n_iters: int = 10000, batch_size: int = 1024, lr: float = 5e-4):
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
            rays_o, rays_d = self.get_rays(pose, self.img_h, self.img_w, self.focal)
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
    
    def extract_mesh(self, resolution: int = 128, threshold: float = 50.0):
        """Extract mesh using marching cubes"""
        try:
            from skimage import measure
            import trimesh
        except ImportError:
            print("\u2717 Please install: pip install scikit-image trimesh")
            return None
        
        print(f"\\nExtracting mesh (resolution: {resolution})...")
        
        # Create grid
        bound = 2.0
        x = torch.linspace(-bound, bound, resolution, device=self.device)
        y = torch.linspace(-bound, bound, resolution, device=self.device)
        z = torch.linspace(-bound, bound, resolution, device=self.device)
        
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        positions = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        
        # Dummy directions (not used for density)
        directions = torch.zeros_like(positions)
        
        # Query density in batches
        batch_size = 10000
        densities = []
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(positions), batch_size):
                batch_pos = positions[i:i+batch_size]
                batch_dir = directions[i:i+batch_size]
                _, density = self.model(batch_pos, batch_dir)
                densities.append(density.cpu())
        
        densities = torch.cat(densities, dim=0)
        density_grid = densities.reshape(resolution, resolution, resolution).numpy()
        
        # Density diagnostics
        print(f"  Density range: [{density_grid.min():.4f}, {density_grid.max():.4f}]")
        print(f"  Density mean: {density_grid.mean():.4f}, std: {density_grid.std():.4f}")
        
        # Adjust threshold if needed
        if density_grid.max() < threshold:
            threshold = density_grid.mean() + 0.5 * density_grid.std()
            print(f"  Adjusted threshold to {threshold:.4f} (original {10.0:.1f} was too high)")
        
        # Marching cubes
        print("  Running marching cubes...")
        try:
            verts, faces, normals, values = measure.marching_cubes(
                density_grid,
                level=threshold,
                spacing=(2*bound/resolution, 2*bound/resolution, 2*bound/resolution)
            )
            
            # Offset vertices
            verts = verts - bound
            
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
        
        # Sample random points in space
        bound = 2.0
        positions = torch.FloatTensor(num_points, 3).uniform_(-bound, bound).to(self.device)
        directions = torch.zeros_like(positions)
        
        # Query density
        self.model.eval()
        with torch.no_grad():
            colors, densities = self.model(positions, directions)
        
        densities_np = densities.squeeze().cpu().numpy()
        print(f"  Density range: [{densities_np.min():.4f}, {densities_np.max():.4f}]")
        print(f"  Density mean: {densities_np.mean():.4f}, std: {densities_np.std():.4f}")
        
        # Adaptive threshold based on actual density distribution
        threshold = densities_np.mean() + 0.5 * densities_np.std()
        print(f"  Using adaptive threshold: {threshold:.4f}")
        
        mask = densities.squeeze() > threshold
        
        points = positions[mask].cpu().numpy()
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
                     num_points: int = 100000):
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
    trainer.train(n_iters=n_iters)
    
    # Extract mesh
    trainer.extract_mesh(resolution=mesh_resolution)
    
    # Extract point cloud
    trainer.extract_pointcloud(num_points=num_points)
    
    print("="*70)
    print("\u2713 Instant NGP reconstruction completed!")
    print("="*70)


if __name__ == "__main__":
    train_simple_nerf(
        data_dir="../reconstructions/InstantNGP_preprocessed/2_segmentation",
        output_dir="../reconstructions/InstantNGP_preprocessed/3_reconstruction",
        n_iters=10000,
        mesh_resolution=128,
        num_points=100000
    )
