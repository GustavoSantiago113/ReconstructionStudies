import os
import struct
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import open3d as o3d

# COLMAP camera model id -> number of float parameters
_CAMERA_MODEL_NUM_PARAMS = {
    0: 3,   # SIMPLE_PINHOLE
    1: 4,   # PINHOLE
    2: 4,   # SIMPLE_RADIAL
    3: 5,   # RADIAL
    4: 8,   # OPENCV
    5: 8,   # OPENCV_FISHEYE
    6: 12,  # FULL_OPENCV
    7: 5,   # FOV
    8: 4,   # SIMPLE_RADIAL_FISHEYE
    9: 5,   # RADIAL_FISHEYE
    10: 12, # THIN_PRISM_FISHEYE
}

_CAMERA_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}


def _parse_cameras_bin(path: str) -> Dict[int, Dict]:
    cams = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            num_params = _CAMERA_MODEL_NUM_PARAMS.get(model_id, 0)
            params = list(struct.unpack(f"<{num_params}d", f.read(8 * num_params)))
            model_name = _CAMERA_MODEL_NAMES.get(model_id, str(model_id))
            cams[cam_id] = {
                "model": model_name,
                "width": width,
                "height": height,
                "params": params,
            }
    return cams


def _parse_images_bin(path: str) -> Dict[int, Dict]:
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            img_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))  # qw qx qy qz
            tvec = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]
            # read null-terminated name
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8")
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            # skip 2D point entries (x, y: double each + point3D_id: int64)
            f.read(num_points2d * 24)
            images[img_id] = {
                "qvec": np.array(qvec, dtype=float),  # qw, qx, qy, qz
                "tvec": np.array(tvec, dtype=float),
                "camera_id": cam_id,
                "name": name,
            }
    return images


def _parse_points3d_bin(path: str) -> Dict[int, Dict]:
    pts = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            pid = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            _error = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)  # skip track (image_id uint32 + point2D_idx uint32)
            pts[pid] = {
                "xyz": np.array(xyz, dtype=float),
                "rgb": np.array(rgb, dtype=np.uint8),
            }
    return pts


def load_colmap_bin_model(model_dir: str) -> Tuple[Dict, Dict, Dict]:
    """Load COLMAP binary-format model files from a folder.

    Expects: ``cameras.bin``, ``images.bin``, ``points3D.bin`` in `model_dir`.

    Returns (cameras, images, points3D) dictionaries.
    """
    model_dir = Path(model_dir)
    cam_f = model_dir / "cameras.bin"
    img_f = model_dir / "images.bin"
    pts_f = model_dir / "points3D.bin"
    if not cam_f.exists() or not img_f.exists() or not pts_f.exists():
        raise FileNotFoundError("COLMAP binary model files not found in " + str(model_dir))
    cams = _parse_cameras_bin(str(cam_f))
    images = _parse_images_bin(str(img_f))
    points = _parse_points3d_bin(str(pts_f))
    return cams, images, points


def load_colmap_model(model_dir: str) -> Tuple[Dict, Dict, Dict]:
    """Auto-detect and load a COLMAP model (binary or text) from `model_dir`.

    Prefers binary format if `cameras.bin` exists, otherwise falls back to text.
    Returns (cameras, images, points3D) dictionaries.
    """
    model_dir = Path(model_dir)
    if (model_dir / "cameras.bin").exists():
        return load_colmap_bin_model(model_dir)
    return load_colmap_text_model(model_dir)


def _parse_cameras_txt(path: str) -> Dict[int, Dict]:
    cams = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            cams[cam_id] = {"model": model, "width": width, "height": height, "params": params}
    return cams


def _parse_images_txt(path: str) -> Dict[int, Dict]:
    images = {}
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        header = lines[i].split()
        # image_id, qx qy qz qw, tx ty tz, camera_id, name
        img_id = int(header[0])
        qx, qy, qz, qw = map(float, header[1:5])
        tx, ty, tz = map(float, header[5:8])
        cam_id = int(header[8])
        name = header[9]
        # next line has 2D points; we don't need it for poses
        images[img_id] = {
            "qvec": np.array([qw, qx, qy, qz], dtype=float),
            "tvec": np.array([tx, ty, tz], dtype=float),
            "camera_id": cam_id,
            "name": name,
        }
        i += 2
    return images


def _parse_points3d_txt(path: str) -> Dict[int, Dict]:
    pts = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pid = int(parts[0])
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            # rest: error, track...
            pts[pid] = {"xyz": np.array([x, y, z], dtype=float), "rgb": np.array([r, g, b], dtype=np.uint8)}
    return pts


def load_colmap_text_model(model_dir: str) -> Tuple[Dict, Dict, Dict]:
    """Load COLMAP text-format model files from a folder.

    Expects: `cameras.txt`, `images.txt`, `points3D.txt` in `model_dir`.

    Returns (cameras, images, points3D) dictionaries.
    """
    model_dir = Path(model_dir)
    cam_f = model_dir / "cameras.txt"
    img_f = model_dir / "images.txt"
    pts_f = model_dir / "points3D.txt"
    if not cam_f.exists() or not img_f.exists() or not pts_f.exists():
        raise FileNotFoundError("COLMAP text model files not found in " + str(model_dir))
    cams = _parse_cameras_txt(str(cam_f))
    images = _parse_images_txt(str(img_f))
    points = _parse_points3d_txt(str(pts_f))
    return cams, images, points


def create_open3d_geometries_from_colmap(
    model_dir: str,
    scale_frustum: float = 0.2,
) -> List[o3d.geometry.Geometry]:
    """Return a list of Open3D geometries from a COLMAP text model folder.

    The first geometry will be the dense point cloud (from points3D.txt),
    followed by camera frustums (LineSets) and a trajectory LineSet.
    """
    cams, images, points = load_colmap_model(model_dir)

    # Build point cloud
    if points:
        pts = np.vstack([v["xyz"] for v in points.values()])
        cols = np.vstack([v["rgb"] for v in points.values()])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector((cols.astype(float) / 255.0))
    else:
        pcd = o3d.geometry.PointCloud()

    geometries: List[o3d.geometry.Geometry] = [pcd]

    # Create frustums and collect camera centers in world coords
    cam_centers = []
    cam_ids_sorted = sorted(images.keys())
    for img_id in cam_ids_sorted:
        info = images[img_id]
        qvec = info["qvec"]  # qw, qx, qy, qz
        tvec = info["tvec"]
        qw, qx, qy, qz = qvec
        # quaternion to rotation matrix (world->camera in COLMAP: qvec rotates from world to camera)
        # convert to rotation matrix R (world->camera), so camera center C = -R.T @ tvec
        q = np.array([qw, qx, qy, qz], dtype=float)
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
            [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
            [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=float)

        C = (-R.T @ tvec).ravel()
        cam_centers.append(C)

        # Build simplistic frustum in world coordinates
        # Determine image size (fall back to sensible defaults)
        cam_id = info["camera_id"]
        cam = cams.get(cam_id, None)
        w_img = cam["width"] if cam is not None else 640
        h_img = cam["height"] if cam is not None else 480

        # Default intrinsics: assume unit focal and principal point at image center
        fx = fy = 1.0
        cx = w_img / 2.0
        cy = h_img / 2.0

        # Override with camera params when available (PINHOLE-like models)
        if cam and cam["model"].upper().startswith("PINHOLE"):
            params = cam.get("params", [])
            if len(params) >= 4:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]

        # frustum corners in camera space
        corners = np.array([[0, 0, 1], [w_img, 0, 1], [w_img, h_img, 1], [0, h_img, 1]], dtype=float)
        # unproject to camera coordinates using approx focal
        corners_cam = np.column_stack(((corners[:, 0] - cx) / fx, (corners[:, 1] - cy) / fy, np.ones(4)))
        corners_cam *= scale_frustum
        # Transform to world: x_world = R.T @ (x_cam) + C
        corners_world = (R.T @ corners_cam.T).T + C.reshape(1, 3)

        points_ls = np.vstack([C.reshape(1, 3), corners_world])
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(points_ls)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0] for _ in lines])
        geometries.append(ls)

    # Trajectory line
    if cam_centers:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.array(cam_centers))
        lines = [[i, i + 1] for i in range(len(cam_centers) - 1)]
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0] for _ in lines])
        geometries.append(line_set)

    return geometries


def save_colmap_geometries_ply(geometries: List[o3d.geometry.Geometry], out_path: str, overwrite: bool = True) -> str:
    """Merge geometries into a point cloud and save as PLY.

    Camera frustums/lines will be sampled as their vertices and colored red.
    """
    out_p = Path(out_path)
    if out_p.exists() and not overwrite:
        raise FileExistsError(out_path + " exists")

    all_pts = []
    all_cols = []
    for g in geometries:
        if isinstance(g, o3d.geometry.PointCloud):
            pts = np.asarray(g.points)
            all_pts.append(pts)
            if g.has_colors():
                cols = (np.asarray(g.colors) * 255.0).astype(np.uint8)
            else:
                cols = np.tile(np.array([[150, 150, 150]], dtype=np.uint8), (len(pts), 1))
            all_cols.append(cols)
        elif isinstance(g, (o3d.geometry.LineSet, o3d.geometry.TriangleMesh)):
            try:
                pts = np.asarray(g.points)
            except Exception:
                pts = np.asarray(g.vertices)
            if pts is None or len(pts) == 0:
                continue
            all_pts.append(pts)
            cols = np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (len(pts), 1))
            all_cols.append(cols)

    if not all_pts:
        raise RuntimeError("No geometry to save")

    pts_concat = np.vstack(all_pts)
    cols_concat = np.vstack(all_cols)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_concat)
    pcd.colors = o3d.utility.Vector3dVector((cols_concat.astype(float) / 255.0))

    out_p.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_p), pcd)
    return str(out_p)
