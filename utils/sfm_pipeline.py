"""
Structure from Motion Pipeline
Incremental reconstruction from feature matches with Open3D visualization.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import open3d as o3d


class IncrementalSfM:
    """
    Incremental Structure from Motion pipeline that reconstructs 3D scene
    from feature matches between multiple images.
    """
    
    def __init__(self, image_size: Tuple[int, int], focal_length: Optional[float] = None):
        """
        Initialize SfM pipeline.
        
        Args:
            image_size: (width, height) of images
            focal_length: Camera focal length (if None, estimate from image size)
        """
        self.image_size = image_size
        w, h = image_size
        
        # Camera intrinsics
        if focal_length is None:
            focal_length = max(w, h) * 1.2
        
        self.K = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Reconstruction state
        self.cameras = {}  # image_path -> {'R', 't', 'P'}
        self.points3d = []  # List of 3D points with colors and tracks
        self.tracks = {}  # track_id -> [(img_path, xy)]
        
    def build_tracks(self, pair_results: List[Dict], max_points_per_pair: int = 3000) -> Dict:
        """
        Build feature tracks from pairwise matches using Union-Find.
        
        Args:
            pair_results: List of pair matching results
            max_points_per_pair: Maximum matches to use per pair
            
        Returns:
            Dictionary of tracks: track_id -> [(img_path, xy)]
        """
        print("Building feature tracks...")
        
        # Union-Find data structure
        parent = {}
        
        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]
        
        def union(i, j):
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                parent[root_i] = root_j
        
        # Map (image_path, x, y) -> unique_id
        obs_to_id = {}
        next_id = 0
        
        def get_key(img_path, xy):
            return (str(img_path), round(xy[0], 1), round(xy[1], 1))
        
        for pair in pair_results:
            img1 = str(pair['img1_path'])
            img2 = str(pair['img2_path'])
            m0 = pair['matches_im0']
            m1 = pair['matches_im1']
            
            # Subsample if needed
            if len(m0) > max_points_per_pair:
                idx = np.random.choice(len(m0), size=max_points_per_pair, replace=False)
                m0, m1 = m0[idx], m1[idx]
            
            for i in range(len(m0)):
                k1 = get_key(img1, m0[i])
                k2 = get_key(img2, m1[i])
                
                if k1 not in obs_to_id:
                    obs_to_id[k1] = next_id
                    parent[next_id] = next_id
                    next_id += 1
                if k2 not in obs_to_id:
                    obs_to_id[k2] = next_id
                    parent[next_id] = next_id
                    next_id += 1
                
                union(obs_to_id[k1], obs_to_id[k2])
        
        # Group observations by track
        tracks = {}
        for key, uid in obs_to_id.items():
            root = find(uid)
            if root not in tracks:
                tracks[root] = []
            tracks[root].append((key[0], np.array([key[1], key[2]])))
        
        # Filter short tracks
        tracks = {tid: obs for tid, obs in tracks.items() 
                 if len(set(o[0] for o in obs)) >= 2}
        
        print(f"  Built {len(tracks)} tracks")
        return tracks
    
    def initialize_reconstruction(self, pair_results: List[Dict]) -> bool:
        """
        Initialize reconstruction with best image pair.
        
        Args:
            pair_results: List of pair matching results
            
        Returns:
            True if initialization succeeded
        """
        print("\nInitializing reconstruction...")
        
        # Select best initial pair
        best_idx = self._select_initial_pair(pair_results)
        if best_idx is None:
            print("  Failed to find suitable initial pair")
            return False
        
        pair = pair_results[best_idx]
        print(f"  Selected: {pair['img1_path'].name} <-> {pair['img2_path'].name}")
        
        # Reconstruct initial pair
        success = self._reconstruct_two_views(
            pair['img1_path'], pair['img2_path'],
            pair['matches_im0'], pair['matches_im1']
        )
        
        if success:
            print(f"  ✓ Initialized with {len(self.points3d)} points")
        return success
    
    def _select_initial_pair(self, pair_results: List[Dict]) -> Optional[int]:
        """Select best initial pair based on matches and baseline."""
        best_idx, best_score = None, -1.0
        
        for idx, pair in enumerate(pair_results):
            m0 = pair['matches_im0']
            if len(m0) < 30:
                continue
            
            # Score based on number of matches and spatial spread
            spread = np.prod(np.std(m0, axis=0) + 1e-6)
            score = len(m0) * spread
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        return best_idx
    
    def _reconstruct_two_views(self, img1_path, img2_path, pts1, pts2) -> bool:
        """Reconstruct initial two views."""
        # Essential matrix decomposition
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, 
                                       prob=0.999, threshold=1.0)
        if E is None:
            return False
        
        inlier_pts1 = pts1[mask.ravel() == 1]
        inlier_pts2 = pts2[mask.ravel() == 1]
        
        # Recover pose
        _, R, t, pose_mask = cv2.recoverPose(E, inlier_pts1, inlier_pts2, self.K)
        
        pose_inlier_pts1 = inlier_pts1[pose_mask.ravel() > 0]
        pose_inlier_pts2 = inlier_pts2[pose_mask.ravel() > 0]
        
        if len(pose_inlier_pts1) < 20:
            return False
        
        # Set up cameras
        img1_str = str(img1_path)
        img2_str = str(img2_path)
        
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t])
        
        self.cameras[img1_str] = {'R': np.eye(3), 't': np.zeros((3, 1)), 'P': P1}
        self.cameras[img2_str] = {'R': R, 't': t, 'P': P2}
        
        # Triangulate points
        pts4d = cv2.triangulatePoints(P1, P2, pose_inlier_pts1.T, pose_inlier_pts2.T)
        pts3d = (pts4d[:3, :] / pts4d[3, :]).T
        
        # Sample colors from images
        try:
            img1_data = cv2.imread(img1_str)
            if img1_data is not None:
                img1_rgb = cv2.cvtColor(img1_data, cv2.COLOR_BGR2RGB)
                colors = []
                for pt in pose_inlier_pts1:
                    x, y = int(round(pt[0])), int(round(pt[1]))
                    h, w = img1_rgb.shape[:2]
                    if 0 <= x < w and 0 <= y < h:
                        colors.append(img1_rgb[y, x])
                    else:
                        colors.append([128, 128, 128])
            else:
                colors = [[128, 128, 128]] * len(pts3d)
        except Exception:
            colors = [[128, 128, 128]] * len(pts3d)
        
        # Store points
        for i, (xyz, color) in enumerate(zip(pts3d, colors)):
            self.points3d.append({
                'xyz': xyz,
                'color': np.array(color, dtype=np.uint8),
                'track_ids': []
            })
        
        return True
    
    def register_images(self, pair_results: List[Dict], 
                       min_obs_for_pnp: int = 10,
                       max_points: int = 50000,
                       pnp_reproj_error: float = 12.0,
                       pnp_min_inliers: int = 8) -> int:
        """
        Register remaining images using PnP.
        
        Args:
            pair_results: List of pair matching results
            min_obs_for_pnp: Minimum 2D-3D correspondences for PnP
            max_points: Maximum total points to prevent memory issues
            
        Returns:
            Number of registered images
        """
        print("\nRegistering additional images...")
        
        all_imgs = set()
        for pair in pair_results:
            all_imgs.add(str(pair['img1_path']))
            all_imgs.add(str(pair['img2_path']))
        
        remaining = list(all_imgs - set(self.cameras.keys()))
        initial_count = len(self.cameras)
        
        # Match tracks to initial points
        self._associate_tracks_to_points(pair_results)
        
        # Incremental registration loop
        iteration = 0
        while remaining and len(self.points3d) < max_points:
            iteration += 1
            best_img = self._find_next_best_view(remaining, min_obs_for_pnp)
            
            if best_img is None:
                break
            
            success = self._register_image(best_img, pnp_reproj_error, pnp_min_inliers)
            if success:
                remaining.remove(best_img)
                # Triangulate new points
                new_points = self._triangulate_new_tracks(best_img, pair_results)
                print(f"  [{iteration}] {Path(best_img).name}: +{new_points} points")
            else:
                break
        
        registered = len(self.cameras) - initial_count
        print(f"\n✓ Registered {registered} additional images")
        print(f"  Total: {len(self.cameras)} cameras, {len(self.points3d)} points")
        
        return registered
    
    def _associate_tracks_to_points(self, pair_results: List[Dict]):
        """Associate track IDs to initial 3D points."""
        for tid, obs in self.tracks.items():
            # Check if track is observed in initial cameras
            obs_in_init = [(img, xy) for img, xy in obs if img in self.cameras]
            if len(obs_in_init) >= 2:
                # Try to match to an existing point (simple proximity check)
                for pt in self.points3d:
                    if len(pt['track_ids']) == 0:  # Unassigned point
                        pt['track_ids'].append(tid)
                        break
    
    def _find_next_best_view(self, candidates: List[str], min_obs: int) -> Optional[str]:
        """Find next best view to register based on visible 3D points."""
        best_img, best_count = None, 0
        
        for img_path in candidates:
            # Count visible points
            count = 0
            for pt in self.points3d:
                for tid in pt['track_ids']:
                    if any(img == img_path for img, _ in self.tracks.get(tid, [])):
                        count += 1
                        break
            
            if count >= min_obs and count > best_count:
                best_count = count
                best_img = img_path
        
        return best_img
    
    def _register_image(self, img_path: str,
                        reprojection_error: float,
                        min_inliers: int) -> bool:
        """Register a new image using PnP RANSAC with tunable thresholds."""
        # Collect 2D-3D correspondences
        pts3d, pts2d = [], []
        for pt in self.points3d:
            for tid in pt['track_ids']:
                for img, xy in self.tracks.get(tid, []):
                    if img == img_path:
                        pts3d.append(pt['xyz'])
                        pts2d.append(xy)
                        break
        
        # Require at least a modest number of correspondences
        if len(pts3d) < max(6, min_inliers):
            return False
        
        pts3d = np.array(pts3d, dtype=np.float64)
        pts2d = np.array(pts2d, dtype=np.float64)
        
        # PnP RANSAC
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d, pts2d, self.K, None,
                iterationsCount=100,
                reprojectionError=reprojection_error,
                confidence=0.999,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        except Exception:
            return False
        
        # Check that we have enough inliers
        if not success or inliers is None or len(inliers) < min_inliers:
            return False
        
        # Store camera
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)
        P = self.K @ np.hstack([R, t])
        
        self.cameras[img_path] = {'R': R, 't': t, 'P': P}
        return True
    
    def _triangulate_new_tracks(self, new_img: str, pair_results: List[Dict]) -> int:
        """Triangulate tracks visible in newly registered image."""
        new_count = 0
        
        for tid, obs in self.tracks.items():
            # Skip if already triangulated
            if any(tid in pt['track_ids'] for pt in self.points3d):
                continue
            
            # Find observations in registered cameras
            obs_reg = [(img, xy) for img, xy in obs if img in self.cameras]
            if len(obs_reg) < 2:
                continue
            
            # Triangulate between new camera and another camera
            obs_new = [xy for img, xy in obs_reg if img == new_img]
            obs_other = [(img, xy) for img, xy in obs_reg if img != new_img]
            
            if len(obs_new) == 0 or len(obs_other) == 0:
                continue
            
            # Use first observation from each
            pt_new = obs_new[0]
            img_other, pt_other = obs_other[0]
            
            # Triangulate
            P1 = self.cameras[img_other]['P']
            P2 = self.cameras[new_img]['P']
            
            pts4d = cv2.triangulatePoints(P1, P2, 
                                         pt_other.reshape(2, 1),
                                         pt_new.reshape(2, 1))
            xyz = (pts4d[:3, :] / pts4d[3, :]).T[0]
            
            # Check reprojection error
            xyz_homo = np.append(xyz, 1.0)
            proj1 = P1 @ xyz_homo
            proj1 = proj1[:2] / proj1[2]
            error1 = np.linalg.norm(proj1 - pt_other)
            
            proj2 = P2 @ xyz_homo
            proj2 = proj2[:2] / proj2[2]
            error2 = np.linalg.norm(proj2 - pt_new)
            
            if error1 < 10.0 and error2 < 10.0:
                # Sample color
                try:
                    img_data = cv2.imread(img_other)
                    if img_data is not None:
                        img_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                        x, y = int(round(pt_other[0])), int(round(pt_other[1]))
                        h, w = img_rgb.shape[:2]
                        if 0 <= x < w and 0 <= y < h:
                            color = img_rgb[y, x]
                        else:
                            color = [128, 128, 128]
                    else:
                        color = [128, 128, 128]
                except Exception:
                    color = [128, 128, 128]
                
                self.points3d.append({
                    'xyz': xyz,
                    'color': np.array(color, dtype=np.uint8),
                    'track_ids': [tid]
                })
                new_count += 1
        
        return new_count
    
    def get_reconstruction_stats(self) -> Dict:
        """Get reconstruction statistics."""
        return {
            'n_cameras': len(self.cameras),
            'n_points': len(self.points3d),
            'n_tracks': len(self.tracks),
            'camera_names': [Path(img).name for img in self.cameras.keys()]
        }
    
    def visualize_with_open3d(self, window_name: str = "SfM Reconstruction"):
        """
        Visualize reconstruction using Open3D.
        
        Args:
            window_name: Window title
        """
        print("\nVisualizing with Open3D...")
        
        geometries = []
        
        # 1. Create point cloud
        if len(self.points3d) > 0:
            points = np.array([pt['xyz'] for pt in self.points3d])
            colors = np.array([pt['color'] for pt in self.points3d]) / 255.0
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(pcd)
            print(f"  Point cloud: {len(points)} points")
        
        # 2. Create camera frustums
        camera_scale = 0.3
        for img_path, cam in self.cameras.items():
            frustum = self._create_camera_frustum(cam, camera_scale)
            geometries.extend(frustum)
        
        print(f"  Cameras: {len(self.cameras)} frustums")
        
        # 3. Add coordinate frame at origin
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        geometries.append(coord_frame)
        
        # Visualize
        print("\n  Opening Open3D viewer...")
        print("  Controls:")
        print("    - Mouse: Rotate view")
        print("    - Scroll: Zoom")
        print("    - Ctrl+Click: Pan")
        print("    - Q: Quit")
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            width=1200,
            height=800,
            point_show_normal=False
        )
    
    def _create_camera_frustum(self, camera: Dict, scale: float = 0.3) -> List:
        """Create camera frustum mesh for visualization."""
        R = camera['R']
        t = camera['t'].reshape(3)
        
        # Camera center in world coordinates
        C = -R.T @ t
        
        # Camera frustum corners (in camera coordinates)
        w, h = self.image_size
        corners_cam = np.array([
            [0, 0, 0],
            [-w/4, -h/4, scale],
            [w/4, -h/4, scale],
            [w/4, h/4, scale],
            [-w/4, h/4, scale]
        ]) / max(w, h)
        
        # Transform to world coordinates
        corners_world = (R.T @ corners_cam.T).T + C
        
        # Create line set for frustum edges
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # From camera center to corners
            [1, 2], [2, 3], [3, 4], [4, 1]   # Rectangle edges
        ]
        
        colors = [[1, 0, 0] for _ in lines]  # Red color for cameras
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners_world)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        # Create sphere at camera center
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere.translate(C)
        sphere.paint_uniform_color([1, 0, 0])
        
        return [line_set, sphere]
    
    def reconstruct(self, pair_results: List[Dict],
                    min_obs_for_pnp: int = 10,
                    pnp_reproj_error: float = 12.0,
                    pnp_min_inliers: int = 8,
                    max_points: int = 50000) -> Dict:
        """
        Full reconstruction pipeline.
        
        Args:
            pair_results: List of pair matching results
            
        Returns:
            Reconstruction statistics
        """
        print("="*60)
        print("INCREMENTAL STRUCTURE FROM MOTION")
        print("="*60)
        
        # Build tracks
        self.tracks = self.build_tracks(pair_results)
        
        # Initialize
        if not self.initialize_reconstruction(pair_results):
            print("\n✗ Reconstruction failed")
            return {'success': False}
        
        # Register images with tunable thresholds
        self.register_images(
            pair_results,
            min_obs_for_pnp=min_obs_for_pnp,
            max_points=max_points,
            pnp_reproj_error=pnp_reproj_error,
            pnp_min_inliers=pnp_min_inliers,
        )
        
        # Get stats
        stats = self.get_reconstruction_stats()
        stats['success'] = True
        
        print("\n" + "="*60)
        print("RECONSTRUCTION COMPLETE")
        print("="*60)
        print(f"Cameras registered: {stats['n_cameras']}")
        print(f"3D points: {stats['n_points']}")
        print(f"Feature tracks: {stats['n_tracks']}")
        print("="*60)
        
        return stats
