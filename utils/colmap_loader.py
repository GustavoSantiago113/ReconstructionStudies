import numpy as np
import struct
from pathlib import Path


def read_cameras_binary(path):
    """
    Read COLMAP cameras.bin file
    
    Returns:
        cameras: dict with camera_id as key and camera params as value
    """
    # COLMAP camera model ID to parameter count mapping
    CAMERA_MODEL_PARAMS = {
        0: 3,   # SIMPLE_PINHOLE (f, cx, cy)
        1: 4,   # PINHOLE (fx, fy, cx, cy)
        2: 4,   # SIMPLE_RADIAL
        3: 5,   # RADIAL
        4: 8,   # OPENCV
        5: 12,  # OPENCV_FISHEYE
        6: 8,   # FULL_OPENCV
        7: 12,  # FOV
        8: 9,   # SIMPLE_RADIAL_FISHEYE
        9: 10,  # RADIAL_FISHEYE
        10: 12, # OPENCV_FISHEYE
        11: 8,  # FOV_FISHEYE
        12: 3,  # THIN_PRISM_FISHEYE
    }
    
    cameras = {}
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise EOFError(f"Unexpected end of file while reading number of cameras from {path}")
        num_cameras = struct.unpack("Q", raw)[0]
        
        for _ in range(num_cameras):
            # Read camera header
            header = f.read(4 + 4 + 8 + 8)  # camera_id (I), model_id (i), width (Q), height (Q)
            if len(header) < (4 + 4 + 8 + 8):
                raise EOFError(f"Unexpected end of file while reading camera header in {path}")
            
            camera_id = struct.unpack_from("I", header, 0)[0]
            model_id = struct.unpack_from("i", header, 4)[0]
            width = struct.unpack_from("Q", header, 8)[0]
            height = struct.unpack_from("Q", header, 16)[0]
            
            # Get number of parameters for this model
            num_params = CAMERA_MODEL_PARAMS.get(model_id, 8)  # Default to 8 if unknown
            
            # Read parameters
            params_bytes = f.read(8 * num_params)
            if len(params_bytes) < 8 * num_params:
                raise EOFError(f"Unexpected end of file while reading {num_params} params for camera {camera_id} (model {model_id}) in {path}")
            params = struct.unpack("d" * num_params, params_bytes)
            
            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params
            }
    
    return cameras


def read_images_binary(path):
    """
    Read COLMAP images.bin file
    
    Returns:
        images: dict with image_id as key and image data as value
    """
    images = {}
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            return images
        num_images = struct.unpack("Q", raw)[0]
        
        for _ in range(num_images):
            image_id = struct.unpack("I", f.read(4))[0]
            
            # Quaternion (qw, qx, qy, qz)
            qvec = struct.unpack("4d", f.read(32))
            
            # Translation (tx, ty, tz)
            tvec = struct.unpack("3d", f.read(24))
            
            camera_id = struct.unpack("I", f.read(4))[0]
            
            # Image name (null-terminated string)
            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if not c or c == b"\x00":
                    break
                name_bytes.extend(c)
            name = name_bytes.decode("utf-8")
            
            # 2D points: skip for now
            num_points2D = struct.unpack("Q", f.read(8))[0]
            # Each point2D entry: x (d), y (d), point3D_id (Q) = 24 bytes
            f.read(24 * num_points2D)
            
            images[image_id] = {
                "qvec": np.array(qvec),
                "tvec": np.array(tvec),
                "camera_id": camera_id,
                "name": name
            }
    
    return images


def read_points3D_binary(path):
    """
    Read COLMAP points3D.bin file
    
    Returns:
        points3D: dict with point_id as key and point data as value
    """
    points3D = {}
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            return points3D
        num_points = struct.unpack("Q", raw)[0]
        
        for _ in range(num_points):
            point_id = struct.unpack("Q", f.read(8))[0]
            xyz = struct.unpack("3d", f.read(24))
            rgb = struct.unpack("3B", f.read(3))
            error = struct.unpack("d", f.read(8))[0]
            track_length = struct.unpack("Q", f.read(8))[0]
            
            # Skip track data (image_id + point2D_idx pairs)
            f.read(8 * track_length)
            
            points3D[point_id] = {
                "xyz": np.array(xyz),
                "rgb": np.array(rgb, dtype=np.uint8),
                "error": error
            }
    
    return points3D


def qvec2rotmat(qvec):
    """Convert quaternion (qw, qx, qy, qz) to rotation matrix"""
    q = np.array(qvec, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])
    
    return R


def get_camera_pose(image_data):
    """
    Convert COLMAP image to camera-to-world pose matrix
    
    Args:
        image_data: dict with 'qvec' and 'tvec'
        
    Returns:
        pose: 4x4 camera-to-world transformation matrix
    """
    R = qvec2rotmat(image_data["qvec"])
    t = image_data["tvec"]
    
    # COLMAP uses world-to-camera, we need camera-to-world
    # c2w = [R^T | -R^T * t]
    pose = np.eye(4)
    pose[:3, :3] = R.T
    pose[:3, 3] = -R.T @ t
    
    return pose


def get_intrinsics(camera):
    """
    Get camera intrinsics matrix
    
    Args:
        camera: dict with camera parameters
        
    Returns:
        K: 3x3 intrinsics matrix
    """
    params = camera["params"]
    model_id = camera["model_id"]
    
    if model_id == 0:  # SIMPLE_PINHOLE
        f, cx, cy = params[:3]
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
    elif model_id == 1:  # PINHOLE
        fx, fy, cx, cy = params[:4]
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    else:
        # For other models, try to extract basic parameters
        try:
            if len(params) >= 4:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif len(params) >= 3:
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                raise ValueError(f"Insufficient parameters for model {model_id}")
            
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
        except Exception as e:
            raise ValueError(f"Unsupported camera model {model_id}: {e}")
    
    return K


class COLMAPDataset:
    """Dataset for loading COLMAP reconstruction data"""
    
    def __init__(self, colmap_dir, image_dir=None, scale=1.0):
        """
        Args:
            colmap_dir: path to COLMAP sparse reconstruction directory
            image_dir: path to images directory (optional)
            scale: scale factor for coordinates
        """
        self.colmap_dir = Path(colmap_dir)
        self.image_dir = Path(image_dir) if image_dir else None
        self.scale = scale
        
        # Read COLMAP binary files
        print("Loading COLMAP data...")
        self.cameras = read_cameras_binary(self.colmap_dir / "cameras.bin")
        self.images = read_images_binary(self.colmap_dir / "images.bin")
        self.points3D = read_points3D_binary(self.colmap_dir / "points3D.bin")
        
        print(f"Loaded {len(self.cameras)} cameras")
        print(f"Loaded {len(self.images)} images")
        print(f"Loaded {len(self.points3D)} 3D points")
        
        # Compute scene bounds
        self._compute_bounds()
        
    def _compute_bounds(self):
        """Compute scene bounding box from 3D points"""
        if len(self.points3D) == 0:
            print("Warning: No 3D points found, using default bounds")
            self.bounds_min = np.array([-1.0, -1.0, -1.0])
            self.bounds_max = np.array([1.0, 1.0, 1.0])
        else:
            points = np.array([p["xyz"] for p in self.points3D.values()])
            self.bounds_min = points.min(axis=0) * self.scale
            self.bounds_max = points.max(axis=0) * self.scale
            
        self.bounds_center = (self.bounds_min + self.bounds_max) / 2
        self.bounds_radius = np.linalg.norm(self.bounds_max - self.bounds_min) / 2
        
        print(f"Scene bounds: min={self.bounds_min}, max={self.bounds_max}")
        print(f"Scene center: {self.bounds_center}, radius: {self.bounds_radius}")
    
    def get_image_data(self, image_id):
        """Get all data for a specific image"""
        image_data = self.images[image_id]
        camera = self.cameras[image_data["camera_id"]]
        
        # Get camera pose and intrinsics
        pose = get_camera_pose(image_data)
        K = get_intrinsics(camera)
        
        # Scale pose
        pose[:3, 3] *= self.scale
        
        return {
            "pose": pose,
            "intrinsics": K,
            "width": camera["width"],
            "height": camera["height"],
            "name": image_data["name"]
        }
    
    def get_all_poses(self):
        """Get all camera poses"""
        poses = []
        for image_id in sorted(self.images.keys()):
            data = self.get_image_data(image_id)
            poses.append(data["pose"])
        return np.array(poses)
    
    def get_train_test_split(self, test_every=8):
        """Split images into train and test sets"""
        image_ids = sorted(self.images.keys())
        
        train_ids = [img_id for i, img_id in enumerate(image_ids) if i % test_every != 0]
        test_ids = [img_id for i, img_id in enumerate(image_ids) if i % test_every == 0]
        
        return train_ids, test_ids
