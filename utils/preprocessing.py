"""
Data Preprocessing Module for NeRF-based 3D Reconstruction

This module implements:
1. Structure from Motion (SfM) for camera pose estimation
2. Pose-Derived Viewing Direction (PDVD) filtering
3. Image Alignment Filtering (IAF) with 20-degree threshold
"""

import os
import numpy as np
import cv2
import json
from pathlib import Path
from typing import List, Tuple, Dict
import subprocess
import shutil


class ImagePreprocessor:
    """
    Preprocesses images for NeRF reconstruction by computing camera poses
    and filtering images based on viewing direction.
    """
    
    def __init__(self, images_dir: str, output_dir: str, angle_threshold: float = 20.0, enable_filtering: bool = True, max_image_size: int = 512):
        """
        Initialize the preprocessor.
        
        Args:
            images_dir: Directory containing input images
            output_dir: Directory to save filtered images and poses
            angle_threshold: Maximum angle in degrees for IAF filtering (default: 20°)
            enable_filtering: Enable IAF filtering (default: True, set False for 360° captures)
            max_image_size: Resize images so largest side is this size (default: 512)
        """
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.angle_threshold = angle_threshold
        self.enable_filtering = enable_filtering
        self.max_image_size = max_image_size
        
        # Create output directories
        self.colmap_dir = self.output_dir / "colmap"
        self.filtered_dir = self.output_dir / "filtered_images"
        self.poses_dir = self.output_dir / "poses"
        
        for dir_path in [self.colmap_dir, self.filtered_dir, self.poses_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def run_opencv_sfm(self) -> bool:
        """
        Run OpenCV-based Structure from Motion to compute camera poses.
        
        Returns:
            True if successful, False otherwise
        """
        print("Running OpenCV-based Structure from Motion...")
        
        try:
            import cv2
            
            # Get all images
            image_files = sorted([f for f in self.images_dir.glob('*.png') if f.is_file()])
            if not image_files:
                image_files = sorted([f for f in self.images_dir.glob('*.jpg') if f.is_file()])
            
            if len(image_files) < 2:
                print(f"✗ Not enough images found: {len(image_files)}")
                return False
            
            print(f"  Found {len(image_files)} images")
            
            # Initialize feature detector (SIFT)
            print("  - Extracting features...")
            sift = cv2.SIFT_create()
            
            # Extract features from all images
            images_data = []
            for img_file in image_files:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kp, desc = sift.detectAndCompute(gray, None)
                images_data.append({
                    'file': img_file,
                    'image': img,
                    'gray': gray,
                    'keypoints': kp,
                    'descriptors': desc
                })
            
            print(f"  - Features extracted from {len(images_data)} images")
            
            # Match features and estimate poses
            print("  - Matching features and estimating poses...")
            poses = self._estimate_poses_opencv(images_data)
            
            # Save poses
            self._save_opencv_poses(poses, images_data)
            
            print("✓ OpenCV SfM completed successfully")
            return True
            
        except ImportError:
            print("✗ OpenCV not found. Please install: pip install opencv-contrib-python")
            return False
        except Exception as e:
            print(f"✗ OpenCV SfM failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _estimate_poses_opencv(self, images_data: List[Dict]) -> Dict[str, Dict]:
        """
        Estimate camera poses using OpenCV sequential matching.
        """
        import cv2
        
        poses = {}
        
        # Estimate camera intrinsics from image size
        h, w = images_data[0]['gray'].shape
        focal_length = max(w, h)
        K = np.array([
            [focal_length, 0, w/2],
            [0, focal_length, h/2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # First camera at origin
        R0 = np.eye(3)
        t0 = np.zeros(3)
        
        poses[images_data[0]['file'].name] = {
            'rotation': R0,
            'translation': t0,
            'camera_center': -R0.T @ t0,
            'quaternion': self._rotmat_to_quat(R0),
            'camera_id': 1,
            'image_id': 1
        }
        
        # Match consecutive images
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        for i in range(len(images_data) - 1):
            desc1 = images_data[i]['descriptors']
            desc2 = images_data[i+1]['descriptors']
            
            if desc1 is None or desc2 is None:
                continue
            
            matches = bf.knnMatch(desc1, desc2, k=2)
            
            # Apply ratio test
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 8:
                # Use previous pose if matching fails
                prev_pose = poses[images_data[i]['file'].name]
                poses[images_data[i+1]['file'].name] = prev_pose.copy()
                poses[images_data[i+1]['file'].name]['image_id'] = i + 2
                continue
            
            # Get matched points
            pts1 = np.float32([images_data[i]['keypoints'][m.queryIdx].pt for m in good_matches])
            pts2 = np.float32([images_data[i+1]['keypoints'][m.trainIdx].pt for m in good_matches])
            
            # Estimate essential matrix
            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            if E is None:
                prev_pose = poses[images_data[i]['file'].name]
                poses[images_data[i+1]['file'].name] = prev_pose.copy()
                poses[images_data[i+1]['file'].name]['image_id'] = i + 2
                continue
            
            # Recover pose
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
            
            # Accumulate transformation
            prev_pose = poses[images_data[i]['file'].name]
            R_prev = prev_pose['rotation']
            t_prev = prev_pose['translation']
            
            R_new = R @ R_prev
            t_new = t_prev + R_prev.T @ t.flatten()
            
            poses[images_data[i+1]['file'].name] = {
                'rotation': R_new,
                'translation': t_new,
                'camera_center': -R_new.T @ t_new,
                'quaternion': self._rotmat_to_quat(R_new),
                'camera_id': 1,
                'image_id': i + 2
            }
        
        return poses
    
    def _save_opencv_poses(self, poses: Dict[str, Dict], images_data: List[Dict]):
        """
        Save OpenCV poses in COLMAP-compatible format.
        """
        sparse_dir = self.colmap_dir / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cameras.txt
        h, w = images_data[0]['gray'].shape
        focal_length = max(w, h)
        
        with open(sparse_dir / "cameras.txt", 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"1 SIMPLE_PINHOLE {w} {h} {focal_length} {w/2} {h/2}\n")
        
        # Save images.txt
        with open(sparse_dir / "images.txt", 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            
            for img_name, pose in poses.items():
                qw, qx, qy, qz = pose['quaternion']
                tx, ty, tz = pose['translation']
                img_id = pose['image_id']
                f.write(f"{img_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {img_name}\n")
                f.write("\n")  # Empty line for POINTS2D
        
        # Save empty points3D.txt
        with open(sparse_dir / "points3D.txt", 'w') as f:
            f.write("# 3D point list\n")
    
    @staticmethod
    def _rotmat_to_quat(R: np.ndarray) -> List[float]:
        """Convert rotation matrix to quaternion (qw, qx, qy, qz)"""
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s
        
        return [qw, qx, qy, qz]
    
    def resize_image(self, image_path: Path, output_path: Path) -> Tuple[int, int, float]:
        """
        Resize image so largest side is max_image_size, maintaining aspect ratio.
        
        Args:
            image_path: Input image path
            output_path: Output image path
            
        Returns:
            (new_width, new_height, scale_factor)
        """
        import cv2
        
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        h, w = img.shape[:2]
        
        # Calculate scale to fit within max_image_size
        scale = self.max_image_size / max(w, h)
        
        if scale >= 1.0:
            # No resizing needed
            shutil.copy2(image_path, output_path)
            return w, h, 1.0
        
        # Resize
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Save
        cv2.imwrite(str(output_path), resized)
        
        return new_w, new_h, scale
    
    def extract_camera_poses(self) -> Dict[str, Dict]:
        """
        Extract camera poses from COLMAP output.
        
        Returns:
            Dictionary mapping image names to pose information
        """
        print("Extracting camera poses...")
        
        sparse_dir = self.colmap_dir / "sparse" / "0"
        images_bin = sparse_dir / "images.bin"
        
        if not images_bin.exists():
            # Try text format
            images_txt = sparse_dir / "images.txt"
            if images_txt.exists():
                return self._read_images_text(images_txt)
            else:
                print("✗ No COLMAP output found")
                return {}
        
        return self._read_images_binary(images_bin)
    
    def _read_images_text(self, filepath: Path) -> Dict[str, Dict]:
        """Read COLMAP images.txt format"""
        poses = {}
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Skip header comments
        lines = [line for line in lines if not line.startswith('#')]
        
        # Parse image entries (every 2 lines)
        for i in range(0, len(lines), 2):
            if i >= len(lines):
                break
            
            parts = lines[i].strip().split()
            if len(parts) < 10:
                continue
            
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            image_name = parts[9]
            
            # Convert quaternion to rotation matrix
            R = self._quat_to_rotmat(qw, qx, qy, qz)
            
            # Camera center: C = -R^T * t
            t = np.array([tx, ty, tz])
            C = -R.T @ t
            
            poses[image_name] = {
                'rotation': R,
                'translation': t,
                'camera_center': C,
                'quaternion': [qw, qx, qy, qz],
                'camera_id': camera_id,
                'image_id': image_id
            }
        
        print(f"✓ Extracted poses for {len(poses)} images")
        return poses
    
    def _read_images_binary(self, filepath: Path) -> Dict[str, Dict]:
        """Read COLMAP images.bin format"""
        import struct
        
        poses = {}
        
        with open(filepath, 'rb') as f:
            num_images = struct.unpack('Q', f.read(8))[0]
            
            for _ in range(num_images):
                image_id = struct.unpack('I', f.read(4))[0]
                qw, qx, qy, qz = struct.unpack('dddd', f.read(32))
                tx, ty, tz = struct.unpack('ddd', f.read(24))
                camera_id = struct.unpack('I', f.read(4))[0]
                
                # Read image name (null-terminated string)
                image_name = b''
                while True:
                    char = f.read(1)
                    if char == b'\x00':
                        break
                    image_name += char
                image_name = image_name.decode('utf-8')
                
                # Skip 2D point data
                num_points2D = struct.unpack('Q', f.read(8))[0]
                f.read(24 * num_points2D)  # Skip x, y, point3D_id for each point
                
                # Convert quaternion to rotation matrix
                R = self._quat_to_rotmat(qw, qx, qy, qz)
                
                # Camera center
                t = np.array([tx, ty, tz])
                C = -R.T @ t
                
                poses[image_name] = {
                    'rotation': R,
                    'translation': t,
                    'camera_center': C,
                    'quaternion': [qw, qx, qy, qz],
                    'camera_id': camera_id,
                    'image_id': image_id
                }
        
        print(f"✓ Extracted poses for {len(poses)} images")
        return poses
    
    @staticmethod
    def _quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        return R
    
    def compute_scene_center(self, poses: Dict[str, Dict]) -> np.ndarray:
        """
        Compute scene center point from camera positions (PDVD).
        
        Args:
            poses: Dictionary of camera poses
            
        Returns:
            3D coordinates of scene center
        """
        camera_centers = np.array([pose['camera_center'] for pose in poses.values()])
        scene_center = np.mean(camera_centers, axis=0)
        
        print(f"✓ Scene center: [{scene_center[0]:.3f}, {scene_center[1]:.3f}, {scene_center[2]:.3f}]")
        return scene_center
    
    def filter_images_by_angle(self, poses: Dict[str, Dict], scene_center: np.ndarray) -> List[str]:
        """
        Filter images using Image Alignment Filtering (IAF).
        Filters out images where the angle between the camera viewing direction
        and the direction to the scene center exceeds the threshold.
        
        Args:
            poses: Dictionary of camera poses
            scene_center: 3D coordinates of scene center
            
        Returns:
            List of image names that pass the filter
        """
        if not self.enable_filtering:
            print("Filtering disabled - keeping all images")
            return list(poses.keys())
        
        print(f"Filtering images (angle threshold: {self.angle_threshold}°)...")
        
        filtered_images = []
        angles = []
        
        for image_name, pose in poses.items():
            # Camera center
            C = pose['camera_center']
            
            # Camera viewing direction (negative z-axis in camera frame)
            R = pose['rotation']
            viewing_dir = -R[2, :]  # Third row of rotation matrix
            
            # Direction from camera to scene center
            to_center_dir = scene_center - C
            to_center_dir = to_center_dir / (np.linalg.norm(to_center_dir) + 1e-10)
            
            # Compute angle
            cos_angle = np.dot(viewing_dir, to_center_dir)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            angles.append(angle_deg)
            
            # Filter based on threshold
            if angle_deg <= self.angle_threshold:
                filtered_images.append(image_name)
        
        print(f"✓ Filtered {len(filtered_images)}/{len(poses)} images")
        print(f"  Angle range: {min(angles):.1f}° - {max(angles):.1f}°")
        print(f"  Mean angle: {np.mean(angles):.1f}°")
        
        # Warn if no images passed
        if len(filtered_images) == 0:
            print(f"\n⚠ WARNING: No images passed filtering!")
            print(f"  For 360° captures, consider:")
            print(f"  1. Disabling filtering: enable_filtering=False")
            print(f"  2. Increasing threshold: angle_threshold=90.0")
        
        return filtered_images
    
    def save_filtered_data(self, filtered_images: List[str], poses: Dict[str, Dict]):
        """
        Save filtered images and their poses.
        
        Args:
            filtered_images: List of image names to keep
            poses: Dictionary of all camera poses
        """
        print("Saving filtered data...")
        
        # Resize and copy filtered images
        image_scales = {}
        final_sizes = {}
        for image_name in filtered_images:
            src = self.images_dir / image_name
            dst = self.filtered_dir / image_name
            if src.exists():
                try:
                    new_w, new_h, scale = self.resize_image(src, dst)
                    image_scales[image_name] = scale
                    final_sizes[image_name] = (new_w, new_h)
                    if scale < 1.0:
                        print(f"  Resized {image_name}: scale={scale:.3f}, size={new_w}x{new_h}")
                except Exception as e:
                    print(f"  Warning: Could not resize {image_name}: {e}")
                    shutil.copy2(src, dst)
                    image_scales[image_name] = 1.0
        
        # Save poses in JSON format
        filtered_poses = {img: poses[img] for img in filtered_images if img in poses}
        
        # Convert numpy arrays to lists for JSON serialization
        for img_name, pose in filtered_poses.items():
            filtered_poses[img_name] = {
                'rotation': pose['rotation'].tolist(),
                'translation': pose['translation'].tolist(),
                'camera_center': pose['camera_center'].tolist(),
                'quaternion': pose['quaternion'],
                'camera_id': pose['camera_id'],
                'image_id': pose['image_id']
            }
        
        poses_file = self.poses_dir / "filtered_poses.json"
        with open(poses_file, 'w') as f:
            json.dump(filtered_poses, f, indent=2)
        
        # Also save in COLMAP-compatible format for Instant-NGP
        self._save_transforms_json(filtered_poses, image_scales, final_sizes)
        
        print(f"✓ Saved {len(filtered_images)} filtered images")
        print(f"  Images: {self.filtered_dir}")
        print(f"  Poses: {poses_file}")
    
    def _save_transforms_json(self, poses: Dict[str, Dict], image_scales: Dict[str, float], final_sizes: Dict[str, Tuple[int, int]]):
        """
        Save poses in transforms.json format compatible with Instant-NGP.
        """
        # Read camera intrinsics from COLMAP
        cameras_file = self.colmap_dir / "sparse" / "0" / "cameras.txt"
        
        fx, fy, cx, cy = 500, 500, 256, 256  # Default values
        w_orig, h_orig = 512, 512
        
        if cameras_file.exists():
            with open(cameras_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        w_orig, h_orig = int(parts[2]), int(parts[3])
                        fx, fy, cx, cy = map(float, parts[4:8])
                        break
        
        # Use actual resized dimensions and scale intrinsics
        if final_sizes:
            # Use the size from the first image (all should be similar after resizing)
            first_img = next(iter(final_sizes.keys()))
            w, h = final_sizes[first_img]
            scale = image_scales[first_img]
            fx, fy, cx, cy = fx * scale, fy * scale, cx * scale, cy * scale
        else:
            w, h = w_orig, h_orig
        
        transforms = {
            "camera_angle_x": 2 * np.arctan(w / (2 * fx)),
            "camera_angle_y": 2 * np.arctan(h / (2 * fy)),
            "fl_x": fx,
            "fl_y": fy,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "frames": []
        }
        
        for img_name, pose in poses.items():
            # Convert from COLMAP to NeRF coordinate system
            R = np.array(pose['rotation'])
            t = np.array(pose['translation'])
            
            # COLMAP uses: X right, Y down, Z forward
            # NeRF uses: X right, Y up, Z backward
            # Transformation matrix
            transform = np.eye(4)
            transform[:3, :3] = R
            transform[:3, 3] = t
            
            # Apply coordinate transformation
            flip_mat = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
            transform = flip_mat @ transform
            
            transforms["frames"].append({
                "file_path": f"./filtered_images/{img_name}",
                "transform_matrix": transform.tolist()
            })
        
        transforms_file = self.output_dir / "transforms.json"
        with open(transforms_file, 'w') as f:
            json.dump(transforms, f, indent=2)
        
        print(f"✓ Saved transforms.json for Instant-NGP")
    
    def run_full_pipeline(self):
        """
        Run the complete preprocessing pipeline:
        1. Run COLMAP SfM
        2. Extract camera poses
        3. Compute scene center (PDVD)
        4. Filter images by angle (IAF)
        5. Save filtered data
        """
        print("="*60)
        print("Starting Image Preprocessing Pipeline")
        print("="*60)
        
        # Step 1: Run SfM
        if not self.run_opencv_sfm():
            print("✗ Pipeline failed at SfM stage")
            return False
        
        # Step 2: Extract poses
        poses = self.extract_camera_poses()
        if not poses:
            print("✗ Pipeline failed: No poses extracted")
            return False
        
        # Step 3: Compute scene center
        scene_center = self.compute_scene_center(poses)
        
        # Step 4: Filter images
        filtered_images = self.filter_images_by_angle(poses, scene_center)
        
        if not filtered_images:
            print("✗ Pipeline failed: No images passed filtering")
            return False
        
        # Step 5: Save results
        self.save_filtered_data(filtered_images, poses)
        
        print("="*60)
        print("✓ Preprocessing pipeline completed successfully!")
        print("="*60)
        
        return True


if __name__ == "__main__":
    # Example usage
    preprocessor = ImagePreprocessor(
        images_dir="../frames",
        output_dir="../reconstructions/InstantNGP_preprocessed",
        angle_threshold=20.0
    )
    
    preprocessor.run_full_pipeline()
