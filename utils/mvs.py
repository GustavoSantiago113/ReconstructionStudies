
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from utils.sfm import SfMReconstruction, create_camera_frustum


# ---------------------------------------------------------------------------
# 10. Multi-View Stereo (MVS)
# ---------------------------------------------------------------------------

def dense_stereo_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    K: np.ndarray,
    R_rel: np.ndarray,
    t_rel: np.ndarray,
    num_disparities: int = 64,
    block_size: int = 7,
    depth_min: float = 0.01,
    depth_max: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense stereo matching for an image pair with known relative pose.

    Parameters
    ----------
    img1, img2     : BGR images sharing the same intrinsics K
    K              : (3,3) intrinsic matrix
    R_rel, t_rel   : relative pose — maps camera-1 coords to camera-2 coords
                     x_cam2 = R_rel @ x_cam1 + t_rel
    num_disparities: SGBM disparity range (rounded to nearest multiple of 16)
    block_size     : SGBM matching block size (forced to odd)
    depth_min/max  : depth range filter in camera-1 units

    Returns
    -------
    pts_cam1 : (N,3) 3-D points in the *original* camera-1 coordinate frame
    colors   : (N,3) uint8 RGB colours sampled from the rectified left image
    disp_vis : (H,W) float32 disparity map for visualisation
    """
    h, w = img1.shape[:2]
    # Enforce SGBM constraints
    num_disparities = max(16, (num_disparities // 16) * 16)
    block_size = block_size if block_size % 2 == 1 else block_size + 1

    t_vec = np.asarray(t_rel).ravel()

    # --- Stereo rectification ---
    R1_rect, R2_rect, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, None, K, None, (w, h),
        R_rel, t_vec,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )

    map1x, map1y = cv2.initUndistortRectifyMap(K, None, R1_rect, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, None, R2_rect, P2, (w, h), cv2.CV_32FC1)

    img1_r = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    img2_r = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

    gray1 = cv2.cvtColor(img1_r, cv2.COLOR_BGR2GRAY) if img1_r.ndim == 3 else img1_r.copy()
    gray2 = cv2.cvtColor(img2_r, cv2.COLOR_BGR2GRAY) if img2_r.ndim == 3 else img2_r.copy()

    # --- Semi-Global Block Matching (SGBM) ---
    stereo = cv2.StereoSGBM_create(
        minDisparity=1,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8  * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = stereo.compute(gray1, gray2).astype(np.float32) / 16.0
    valid = disp > 1.0

    # --- Reproject to 3-D in the *rectified* camera-1 frame ---
    pts_rect = cv2.reprojectImageTo3D(disp, Q)  # (H, W, 3)

    # --- Un-rectify: rectified cam-1 → original cam-1 ---
    # R1_rect maps original→rectified, so original = rect @ R1_rect (row vectors)
    pts_cam1_rect = pts_rect[valid]            # (N, 3) in rectified frame
    pts_cam1 = pts_cam1_rect @ R1_rect         # (N, 3) in original cam-1 frame

    # --- Depth filter ---
    depths = pts_cam1[:, 2]
    depth_ok = (depths > depth_min) & (depths < depth_max)
    pts_cam1 = pts_cam1[depth_ok]

    # --- Sample colours from the rectified left image ---
    ys, xs = np.where(valid)
    ys = ys[depth_ok]
    xs = xs[depth_ok]
    if img1_r.ndim == 3:
        colors = img1_r[ys, xs][:, ::-1]   # BGR → RGB
    else:
        c = img1_r[ys, xs]
        colors = np.stack([c, c, c], axis=1)

    disp_vis = np.where(valid, disp, 0.0).astype(np.float32)
    return pts_cam1, colors.astype(np.uint8), disp_vis


def run_mvs(
    recon: "SfMReconstruction",
    images: List[np.ndarray],
    pairs: Optional[List[Tuple[int, int]]] = None,
    num_disparities: int = 64,
    block_size: int = 7,
    depth_scale: float = 3.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run dense MVS over all consecutive registered camera pairs.

    For each pair the left image gets a dense depth map via SGBM.  The depth
    map is back-projected to the original camera frame and then to world space
    using the SfM camera poses.

    Parameters
    ----------
    recon           : SfMReconstruction from the SfM + BA step
    images          : list of BGR images indexed by camera id
    pairs           : list of (cid_a, cid_b) to process; if None uses all
                      consecutive registered camera pairs
    num_disparities : SGBM disparity range (rounded up to multiple of 16)
    block_size      : SGBM block size (odd integer)
    depth_scale     : depth_max = 98th-percentile sparse depth × depth_scale
    verbose         : print per-pair progress

    Returns
    -------
    pts_world : (N, 3) float64 dense world-space 3-D points
    colors    : (N, 3) uint8 RGB colours
    """
    cam_ids = sorted(recon.camera_poses.keys())
    if pairs is None:
        pairs = [(cam_ids[i], cam_ids[i + 1]) for i in range(len(cam_ids) - 1)]

    # Per-camera depth bounds derived from the sparse SfM cloud
    sparse_pts = recon.get_points_array()

    def _depth_bounds(cid: int) -> Tuple[float, float]:
        R, t = recon.camera_poses[cid]
        if len(sparse_pts) == 0:
            return 0.01, 100.0
        pts_cam = sparse_pts @ R.T + t.ravel()
        depths = pts_cam[:, 2]
        depths = depths[depths > 0]
        if len(depths) == 0:
            return 0.01, 100.0
        return float(np.percentile(depths, 2)), float(np.percentile(depths, 98) * depth_scale)

    all_pts: List[np.ndarray] = []
    all_cols: List[np.ndarray] = []

    for cid1, cid2 in pairs:
        if cid1 not in recon.camera_poses or cid2 not in recon.camera_poses:
            continue
        if cid1 >= len(images) or cid2 >= len(images):
            continue

        R1, t1 = recon.camera_poses[cid1]
        R2, t2 = recon.camera_poses[cid2]

        # Relative pose: x_cam2 = R_rel @ x_cam1 + t_rel
        R_rel = R2 @ R1.T
        t_rel = (t2.ravel() - R_rel @ t1.ravel()).reshape(3, 1)

        baseline = float(np.linalg.norm(t_rel))
        if baseline < 1e-4:
            if verbose:
                print(f"  Pair ({cid1},{cid2}): skipped – zero baseline")
            continue

        depth_min, depth_max = _depth_bounds(cid1)

        try:
            pts_cam1, cols, _ = dense_stereo_pair(
                images[cid1], images[cid2], recon.K, R_rel, t_rel,
                num_disparities=num_disparities,
                block_size=block_size,
                depth_min=depth_min,
                depth_max=depth_max,
            )
        except cv2.error as e:
            if verbose:
                print(f"  Pair ({cid1},{cid2}): stereo failed – {e}")
            continue

        # cam-1 frame → world frame
        # world-to-cam: x_cam = R1 @ x_world + t1
        # cam-to-world (row vectors): x_world = (x_cam - t1) @ R1
        pts_world_pair = (pts_cam1 - t1.ravel()) @ R1
        all_pts.append(pts_world_pair)
        all_cols.append(cols)

        if verbose:
            print(f"  Pair ({cid1},{cid2}): {len(pts_world_pair):>8,} dense pts  "
                  f"(baseline={baseline:.4f}, depth [{depth_min:.3f}, {depth_max:.3f}])")

    if not all_pts:
        if verbose:
            print("No dense points generated.")
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)

    pts_world  = np.concatenate(all_pts,  axis=0)
    colors_all = np.concatenate(all_cols, axis=0)

    if verbose:
        print(f"\nTotal dense points (before filtering): {len(pts_world):,}")
    return pts_world, colors_all


def filter_dense_cloud(
    pts: np.ndarray,
    colors: Optional[np.ndarray] = None,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Statistical outlier removal for a dense point cloud.

    Each point is kept only when its mean distance to its nearest
    ``nb_neighbors`` neighbours is within ``std_ratio`` standard deviations
    of the global mean distance.
    """
    if len(pts) == 0:
        return pts, colors

    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    # k+1 because the first neighbour is the point itself (distance = 0)
    dists, _ = tree.query(pts, k=nb_neighbors + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    mu  = mean_dists.mean()
    sig = mean_dists.std()
    keep = mean_dists < (mu + std_ratio * sig)
    filtered_cols = colors[keep] if colors is not None else None
    return pts[keep], filtered_cols


def plot_dense_reconstruction(
    pts_dense: np.ndarray,
    colors_dense: Optional[np.ndarray] = None,
    recon: Optional["SfMReconstruction"] = None,
    title: str = "MVS Dense Reconstruction",
    subsample: int = 100000,
    window_name: str = "Dense Reconstruction",
) -> None:
    """Visualize dense MVS point cloud with camera positions using Open3D.
    
    Parameters
    ----------
    pts_dense : (N, 3) dense world-space 3D points
    colors_dense : (N, 3) uint8 RGB colors (optional)
    recon : SfMReconstruction object for camera visualization (optional)
    title : description for console output
    subsample : maximum number of points to display
    window_name : Open3D window name
    """
    if len(pts_dense) == 0:
        print("No dense points available for visualization.")
        return

    # Subsample and filter outliers
    if len(pts_dense) > subsample:
        idx = np.random.choice(len(pts_dense), subsample, replace=False)
        pts_plot = pts_dense[idx]
        cols_plot = colors_dense[idx] if colors_dense is not None else None
    else:
        pts_plot = pts_dense.copy()
        cols_plot = colors_dense.copy() if colors_dense is not None else None

    # Clip display outliers using percentile
    keep = np.ones(len(pts_plot), dtype=bool)
    for ax_i in range(3):
        lo, hi = np.percentile(pts_plot[:, ax_i], [1, 99])
        keep &= (pts_plot[:, ax_i] >= lo) & (pts_plot[:, ax_i] <= hi)
    pts_plot = pts_plot[keep]
    cols_plot = cols_plot[keep] if cols_plot is not None else None

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_plot)
    
    if cols_plot is not None:
        # Normalize RGB colors to 0-1 range
        pcd.colors = o3d.utility.Vector3dVector(cols_plot.astype(np.float64) / 255.0)
    else:
        # Use height-based coloring if no RGB available
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
    geometries = [pcd]
    
    # Add SfM cameras if reconstruction provided
    if recon is not None:
        cam_ids = sorted(recon.camera_poses.keys())
        for cid in cam_ids:
            R, t = recon.camera_poses[cid]
            frustum = create_camera_frustum(R, t, recon.K, scale=0.15, color=(1.0, 0.0, 0.0))
            geometries.append(frustum)
        
        # Add camera trajectory
        if len(cam_ids) > 1:
            cam_centers = recon.get_camera_centers()
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(cam_centers)
            lines = [[i, i+1] for i in range(len(cam_centers)-1)]
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0] for _ in lines])
            geometries.append(line_set)
    
    # Add coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries.append(coord_frame)
    
    # Visualize
    print(f"{title}: Displaying {len(pts_plot):,} points (subsampled from {len(pts_dense):,})")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=1400,
        height=900,
        point_show_normal=False,
    )