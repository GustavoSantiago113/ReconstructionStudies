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


class NeRFReconstructor:
    """
    Handles 3D reconstruction using a NeRF model.
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
                               preview_every: int = 1000,
                               preview_index: int = 0) -> bool:
        """
        Run complete reconstruction pipeline.
        
        Args:
            max_iterations: Training iterations
            mesh_resolution: Marching cubes resolution
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
                preview_every=preview_every,
                preview_index=preview_index
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

if __name__ == "__main__":
    # Example usage
    reconstructor = NeRFReconstructor(
        data_dir="../reconstructions/InstantNGP_preprocessed",
        output_dir="../reconstructions/InstantNGP_output",
    )
    
    reconstructor.run_full_reconstruction(
        max_iterations=30000,
        mesh_resolution=1024,
        preview_every=1000,
        preview_index=0
    )
