"""
Instant-NGP 3D Reconstruction Module

This module handles:
1. Training NeRF models using Instant-NGP
2. Extracting point clouds using marching cubes
3. Mesh and point cloud export
"""

import os
import json
import numpy as np
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict
import shutil

# Import simple NeRF as alternative
try:
    from utils.simple_nerf import train_simple_nerf
except:
    from simple_nerf import train_simple_nerf


class InstantNGPReconstructor:
    """
    Handles 3D reconstruction using Instant-NGP (instant-ngp or nerfstudio implementation).
    """
    
    def __init__(self, 
                 data_dir: str,
                 output_dir: str):
        """
        Initialize the reconstructor.
        
        Args:
            data_dir: Directory containing images and transforms.json
            output_dir: Directory to save reconstruction outputs
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if transforms.json exists
        self.transforms_file = self.data_dir / "transforms.json"
        if not self.transforms_file.exists():
            raise FileNotFoundError(f"transforms.json not found in {data_dir}")
        
        print(f"✓ Initialized reconstructor for {data_dir}")
        print(f"  Using: Simple PyTorch NeRF")
    
    def run_full_reconstruction(self,
                               max_iterations: int = 10000,
                               mesh_resolution: int = 128,
                               num_points: int = 100000,
                               method: str = "simple-nerf") -> bool:
        """
        Run complete reconstruction pipeline.
        
        Args:
            max_iterations: Training iterations
            mesh_resolution: Marching cubes resolution
            num_points: Number of points in point cloud
            method: NeRF method to use ("simple-nerf" or nerfstudio methods)
            
        Returns:
            True if successful
        """
        print("="*60)
        print("Starting 3D Reconstruction")
        print("="*60)
        
        # Use simple PyTorch NeRF
        print("Using Simple PyTorch NeRF implementation...")
        try:
            train_simple_nerf(
                data_dir=str(self.data_dir),
                output_dir=str(self.output_dir),
                n_iters=max_iterations,
                mesh_resolution=mesh_resolution,
                num_points=num_points
            )
        except Exception as e:
            print(f"✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("="*60)
        print("✓ 3D Reconstruction completed!")
        print("="*60)
        
        return True


class MarchingCubesExtractor:
    """
    Extract meshes from density fields using marching cubes algorithm.
    """
    
    @staticmethod
    def extract_mesh_from_density(density_func,
                                  bounds: Tuple[float, float, float],
                                  resolution: int = 256,
                                  threshold: float = 10.0,
                                  output_path: Optional[str] = None):
        """
        Extract mesh using marching cubes.
        
        Args:
            density_func: Function that takes (N, 3) positions and returns (N,) densities
            bounds: (min, max) bounds for each axis
            resolution: Grid resolution
            threshold: Density threshold for isosurface
            output_path: Optional path to save mesh
            
        Returns:
            mesh: Trimesh object
        """
        try:
            import trimesh
            from skimage import measure
            
            print(f"Extracting mesh with marching cubes (resolution: {resolution})...")
            
            # Create grid
            x = np.linspace(-bounds[0], bounds[0], resolution)
            y = np.linspace(-bounds[1], bounds[1], resolution)
            z = np.linspace(-bounds[2], bounds[2], resolution)
            
            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
            positions = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
            
            # Evaluate density
            print("  Evaluating density field...")
            densities = density_func(positions)
            density_grid = densities.reshape(resolution, resolution, resolution)
            
            # Run marching cubes
            print("  Running marching cubes...")
            verts, faces, normals, values = measure.marching_cubes(
                density_grid,
                level=threshold,
                spacing=(2*bounds[0]/resolution, 
                        2*bounds[1]/resolution, 
                        2*bounds[2]/resolution)
            )
            
            # Offset vertices to correct position
            verts = verts - np.array([bounds[0], bounds[1], bounds[2]])
            
            # Create mesh
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
            
            # Save if requested
            if output_path:
                mesh.export(output_path)
                print(f"✓ Mesh saved to {output_path}")
            
            return mesh
            
        except ImportError:
            print("✗ Required packages not installed:")
            print("  pip install trimesh scikit-image")
            return None
        except Exception as e:
            print(f"✗ Error in marching cubes: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    reconstructor = InstantNGPReconstructor(
        data_dir="../reconstructions/InstantNGP_preprocessed",
        output_dir="../reconstructions/InstantNGP_output",
    )
    
    reconstructor.run_full_reconstruction(
        max_iterations=30000,
        mesh_resolution=1024,
        num_points=1000000,
        method="instant-ngp"
    )
