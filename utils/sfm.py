"""
Structure from Motion (SfM) pipeline utilities.

Steps:
  1. Feature Detection & Description  (SIFT / ORB / AKAZE)
  2. Feature Matching                 (FLANN or BF + Lowe ratio test)
  3. Geometric Verification           (Fundamental / Essential via RANSAC)
  4. Camera Intrinsics                (from EXIF or a rough estimate)
  5. Relative Pose Recovery           (Essential matrix decomposition)
  6. Triangulation                    (DLT linear triangulation)
  7. Incremental Reconstruction       (PnP + triangulation)
  8. Bundle Adjustment                (Levenberg-Marquardt via scipy.optimize)
  9. Visualization helpers
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

matplotlib.rcParams["figure.dpi"] = 100

# ---------------------------------------------------------------------------
# 1. Feature Detection & Description
# ---------------------------------------------------------------------------

def load_image(path: str, gray: bool = False) -> np.ndarray:
    """Load an image from disk. Returns BGR (or grayscale) uint8 array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def detect_and_describe(
    image: np.ndarray,
    method: str = "SIFT",
    max_features: int = 4000,
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """Detect keypoints and compute descriptors.

    Parameters
    ----------
    image : BGR or grayscale uint8 array
    method : 'SIFT' | 'ORB' | 'AKAZE'
    max_features : upper bound on returned keypoints

    Returns
    -------
    keypoints : list of cv2.KeyPoint
    descriptors : float32/uint8 array (N, D)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    method = method.upper()
    if method == "SIFT":
        detector = cv2.SIFT_create(nfeatures=max_features)
    elif method == "ORB":
        detector = cv2.ORB_create(nfeatures=max_features)
    elif method == "AKAZE":
        detector = cv2.AKAZE_create()
    else:
        raise ValueError(f"Unknown method '{method}'. Choose SIFT, ORB or AKAZE.")

    kp, des = detector.detectAndCompute(gray, None)
    return kp, des


# ---------------------------------------------------------------------------
# 2. Feature Matching
# ---------------------------------------------------------------------------

def match_features(
    kp1: List[cv2.KeyPoint],
    des1: np.ndarray,
    kp2: List[cv2.KeyPoint],
    des2: np.ndarray,
    method: str = "FLANN",
    ratio: float = 0.75,
) -> Tuple[List[cv2.DMatch], np.ndarray, np.ndarray]:
    """Match descriptors and apply Lowe's ratio test.

    Parameters
    ----------
    method : 'FLANN' (float descriptors) | 'BF' (any descriptor)
    ratio  : Lowe ratio threshold

    Returns
    -------
    good_matches : list of cv2.DMatch (after ratio test)
    pts1, pts2   : (N,2) float32 arrays of matched keypoint coordinates
    """
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return [], np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    # Choose matcher
    if method.upper() == "FLANN":
        if des1.dtype != np.float32:
            des1 = des1.astype(np.float32)
            des2 = des2.astype(np.float32)
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=100)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        norm = cv2.NORM_HAMMING if des1.dtype == np.uint8 else cv2.NORM_L2
        matcher = cv2.BFMatcher(norm)

    matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

    if not good:
        return [], np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return good, pts1, pts2


# ---------------------------------------------------------------------------
# 3. Geometric Verification
# ---------------------------------------------------------------------------

def geometric_verification(
    pts1: np.ndarray,
    pts2: np.ndarray,
    mode: str = "fundamental",
    reproj_threshold: float = 3.0,
    confidence: float = 0.999,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate Fundamental or Homography matrix with RANSAC.

    Parameters
    ----------
    mode : 'fundamental' | 'homography'

    Returns
    -------
    matrix : (3,3) estimated matrix (F or H)
    mask   : (N,1) uint8 inlier mask
    """
    if len(pts1) < 8:
        raise ValueError("Need at least 8 point correspondences for geometric verification.")

    if mode == "fundamental":
        matrix, mask = cv2.findFundamentalMat(
            pts1, pts2,
            cv2.FM_RANSAC,
            ransacReprojThreshold=reproj_threshold,
            confidence=confidence,
        )
    elif mode == "homography":
        matrix, mask = cv2.findHomography(
            pts1, pts2,
            cv2.RANSAC,
            ransacReprojThreshold=reproj_threshold,
            confidence=confidence,
        )
    else:
        raise ValueError("mode must be 'fundamental' or 'homography'")

    if mask is None:
        mask = np.zeros((len(pts1), 1), dtype=np.uint8)

    return matrix, mask.ravel().astype(bool)


# ---------------------------------------------------------------------------
# 4. Camera Intrinsics
# ---------------------------------------------------------------------------

def estimate_intrinsics(
    image_shape: Tuple[int, int],
    fov_deg: float = 60.0,
) -> np.ndarray:
    """Estimate a pinhole camera intrinsic matrix K.

    Uses a single focal-length estimate from horizontal FOV.

    Parameters
    ----------
    image_shape : (height, width) in pixels
    fov_deg     : horizontal field of view in degrees

    Returns
    -------
    K : (3,3) float64 intrinsic matrix
    """
    h, w = image_shape[:2]
    f = (w / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
    cx, cy = w / 2.0, h / 2.0
    K = np.array([
        [f,  0, cx],
        [0,  f, cy],
        [0,  0,  1],
    ], dtype=np.float64)
    return K


# ---------------------------------------------------------------------------
# 5. Essential Matrix & Relative Pose
# ---------------------------------------------------------------------------

def compute_essential(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    reproj_threshold: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute essential matrix with RANSAC.

    Returns
    -------
    E    : (3,3) essential matrix
    mask : (N,) boolean inlier mask
    """
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=reproj_threshold,
    )
    if mask is None:
        mask = np.zeros(len(pts1), dtype=bool)
    return E, mask.ravel().astype(bool)


def recover_pose(
    E: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Decompose essential matrix into R, t.

    Returns
    -------
    R       : (3,3) rotation matrix
    t       : (3,1) translation vector (unit norm)
    n_inliers: number of points in front of both cameras
    """
    if mask is not None:
        pts1_in = pts1[mask]
        pts2_in = pts2[mask]
    else:
        pts1_in, pts2_in = pts1, pts2

    n_inliers, R, t, pose_mask = cv2.recoverPose(E, pts1_in, pts2_in, K)
    return R, t, n_inliers


# ---------------------------------------------------------------------------
# 6. Triangulation
# ---------------------------------------------------------------------------

def build_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 3x4 projection matrix  P = K [R | t]."""
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt


def triangulate_points(
    P1: np.ndarray,
    P2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    """Triangulate 3-D points from two views using DLT (OpenCV).

    Parameters
    ----------
    P1, P2 : (3,4) projection matrices
    pts1, pts2 : (N,2) matched image points

    Returns
    -------
    pts3d : (N,3) float64 array of triangulated 3-D points
    """
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T  # homogeneous → Euclidean
    return pts3d


def filter_triangulated(
    pts3d: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    max_reproj_error: float = 4.0,
    min_depth: float = 0.0,
) -> np.ndarray:
    """Return boolean mask of well-triangulated points.

    Criteria:
     - Positive depth in both cameras
     - Reprojection error below threshold in both views
    """
    def reproject(P, pts3d):
        ph = np.hstack([pts3d, np.ones((len(pts3d), 1))])  # (N,4)
        proj = (P @ ph.T).T  # (N,3)
        return proj[:, :2] / proj[:, 2:3], proj[:, 2]

    # Positive depth check
    reproj1, d1 = reproject(P1, pts3d)
    reproj2, d2 = reproject(P2, pts3d)
    positive_depth = (d1 > min_depth) & (d2 > min_depth)

    # We need the original 2D points to compute reprojection error —
    # this function returns a positive-depth mask; caller can combine.
    return positive_depth


# ---------------------------------------------------------------------------
# 7. Incremental SfM
# ---------------------------------------------------------------------------

class SfMReconstruction:
    """Holds the incremental SfM state."""

    def __init__(self, K: np.ndarray):
        self.K = K
        # camera_poses[i] = (R, t) for image i
        self.camera_poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        # points3d[track_id] = (X, Y, Z)
        self.points3d: Dict[int, np.ndarray] = {}
        # observations: track_id -> [(img_idx, (u, v)), ...]
        self.observations: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        self._next_track = 0

    def add_camera(self, idx: int, R: np.ndarray, t: np.ndarray):
        self.camera_poses[idx] = (R, t.reshape(3, 1))

    def add_points(
        self,
        pts3d: np.ndarray,
        img_idx1: int,
        img_idx2: int,
        pts2d_1: np.ndarray,
        pts2d_2: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ):
        if mask is not None:
            pts3d = pts3d[mask]
            pts2d_1 = pts2d_1[mask]
            pts2d_2 = pts2d_2[mask]

        for i, pt in enumerate(pts3d):
            tid = self._next_track
            self.points3d[tid] = pt
            self.observations[tid] = [
                (img_idx1, pts2d_1[i]),
                (img_idx2, pts2d_2[i]),
            ]
            self._next_track += 1

    def get_points_array(self) -> np.ndarray:
        """Return (N,3) array of all 3-D points."""
        if not self.points3d:
            return np.empty((0, 3))
        return np.array(list(self.points3d.values()))

    def get_camera_centers(self) -> np.ndarray:
        """Return (M,3) array of camera centres in world coordinates."""
        centers = []
        for R, t in self.camera_poses.values():
            C = -R.T @ t.reshape(3, 1)
            centers.append(C.ravel())
        return np.array(centers) if centers else np.empty((0, 3))

    def get_points_and_colors(
        self, images: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (N,3) points and (N,3) uint8 RGB colors sampled from images.

        For each 3-D point the pixel colour is taken from its first observation
        whose image index is valid.  Points with no valid observation get grey.
        """
        track_ids = sorted(self.points3d.keys())
        pts, colors = [], []
        for tid in track_ids:
            pts.append(self.points3d[tid])
            color = np.array([128, 128, 128], dtype=np.uint8)
            for img_idx, uv in self.observations.get(tid, []):
                if img_idx < len(images):
                    img = images[img_idx]
                    h, w = img.shape[:2]
                    u = int(np.clip(round(uv[0]), 0, w - 1))
                    v = int(np.clip(round(uv[1]), 0, h - 1))
                    bgr = img[v, u]
                    color = np.array([bgr[2], bgr[1], bgr[0]], dtype=np.uint8)
                    break
            colors.append(color)
        if not pts:
            return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
        return np.array(pts), np.array(colors, dtype=np.uint8)


def run_sfm(
    image_paths: List[str],
    K: Optional[np.ndarray] = None,
    feature_method: str = "SIFT",
    match_ratio: float = 0.75,
    reproj_threshold: float = 4.0,
    verbose: bool = True,
) -> SfMReconstruction:
    """Run a minimal incremental SfM pipeline on a list of images.

    1. Detect features in every image.
    2. Match every consecutive pair.
    3. Use the best pair as the initial reconstruction.
    4. Incrementally add remaining cameras with PnP.

    Parameters
    ----------
    image_paths    : ordered list of image file paths
    K              : (3,3) intrinsic matrix; estimated from image if None
    feature_method : 'SIFT' | 'ORB' | 'AKAZE'
    match_ratio    : Lowe ratio test threshold
    reproj_threshold : RANSAC reprojection threshold (pixels)

    Returns
    -------
    SfMReconstruction object
    """
    n = len(image_paths)
    if n < 2:
        raise ValueError("Need at least 2 images.")

    # --- load images & detect features ---
    images, keypoints, descriptors = [], [], []
    for path in image_paths:
        img = load_image(str(path))
        kp, des = detect_and_describe(img, method=feature_method)
        images.append(img)
        keypoints.append(kp)
        descriptors.append(des)
        if verbose:
            print(f"  [{Path(path).name}] {len(kp)} keypoints")

    # --- estimate K if not provided ---
    if K is None:
        K = estimate_intrinsics(images[0].shape)
        if verbose:
            print(f"\nEstimated K (fov=60°):\n{K}\n")

    recon = SfMReconstruction(K)

    # --- initial pair: images 0 and 1 ---
    if verbose:
        print("=== Initial pair: images 0 and 1 ===")
    good, pts1, pts2 = match_features(
        keypoints[0], descriptors[0],
        keypoints[1], descriptors[1],
        ratio=match_ratio,
    )
    if len(pts1) < 8:
        raise RuntimeError("Not enough matches for the initial pair.")

    E, emask = compute_essential(pts1, pts2, K, reproj_threshold=1.5)
    R, t, _ = recover_pose(E, pts1, pts2, K, mask=emask)

    R0 = np.eye(3)
    t0 = np.zeros((3, 1))
    recon.add_camera(0, R0, t0)
    recon.add_camera(1, R, t)

    P0 = build_projection_matrix(K, R0, t0)
    P1 = build_projection_matrix(K, R, t)

    pts1_in = pts1[emask]
    pts2_in = pts2[emask]
    pts3d = triangulate_points(P0, P1, pts1_in, pts2_in)
    depth_mask = filter_triangulated(pts3d, P0, P1)
    recon.add_points(pts3d, 0, 1, pts1_in, pts2_in, mask=depth_mask)

    if verbose:
        print(f"  Triangulated {depth_mask.sum()} initial 3-D points.\n")

    # --- incrementally add remaining cameras ---
    for i in range(2, n):
        if verbose:
            print(f"=== Adding image {i} ===")

        registered = sorted(recon.camera_poses.keys())

        # Build a 2D-3D correspondence set for PnP:
        # For each registered camera ref, match ref <-> i, then for each matched
        # keypoint in ref look up whether it sits near a tracked 3-D point.
        obs_index: Dict[int, Tuple[np.ndarray, List[int]]] = {}
        for ref in registered:
            obs_in_ref = [
                (uv, tid)
                for tid, obs_list in recon.observations.items()
                for (cid, uv) in obs_list
                if cid == ref and tid in recon.points3d
            ]
            if obs_in_ref:
                uvs = np.array([x[0] for x in obs_in_ref], dtype=np.float32)
                tids = [x[1] for x in obs_in_ref]
                obs_index[ref] = (uvs, tids)

        pts3d_pnp: List[np.ndarray] = []
        pts2d_pnp: List[np.ndarray] = []

        best_ref = registered[-1]          # fallback for triangulation below
        best_ref_n = 0

        for ref in registered:
            _, pts_ref_m, pts_i_m = match_features(
                keypoints[ref], descriptors[ref],
                keypoints[i], descriptors[i],
                ratio=match_ratio,
            )
            if len(pts_ref_m) == 0:
                continue

            # Track reference camera with most matches (for triangulation below)
            if len(pts_ref_m) > best_ref_n:
                best_ref_n = len(pts_ref_m)
                best_ref = ref

            # Find 2D-3D correspondences via proximity of matched kp to track obs
            if ref not in obs_index:
                continue
            tracked_uvs, tracked_tids = obs_index[ref]

            for k, pt_ref in enumerate(pts_ref_m):
                dists = np.linalg.norm(tracked_uvs - pt_ref, axis=1)
                j_min = int(np.argmin(dists))
                if dists[j_min] < 3.0:          # within 3 px → same keypoint
                    tid = tracked_tids[j_min]
                    pt3d = recon.points3d[tid]
                    if np.all(np.isfinite(pt3d)):
                        pts3d_pnp.append(pt3d)
                        pts2d_pnp.append(pts_i_m[k])

        if len(pts3d_pnp) < 6:
            if verbose:
                print(f"  Skipping image {i}: only {len(pts3d_pnp)} 2D-3D correspondences "
                      f"(need ≥6).")
            continue

        pts3d_arr = np.array(pts3d_pnp, dtype=np.float64)
        pts2d_arr = np.array(pts2d_pnp, dtype=np.float64)

        # Deduplicate (same 3-D point matched via multiple ref cameras)
        _, uniq_idx = np.unique(pts3d_arr, axis=0, return_index=True)
        pts3d_arr = pts3d_arr[uniq_idx]
        pts2d_arr = pts2d_arr[uniq_idx]

        success, rvec, tvec, _ = cv2.solvePnPRansac(
            pts3d_arr.astype(np.float32),
            pts2d_arr.astype(np.float32),
            K, None,
            iterationsCount=200,
            reprojectionError=reproj_threshold,
            confidence=0.999,
        )
        if not success:
            if verbose:
                print(f"  PnP failed for image {i}. Skipping.")
            continue

        R_i, _ = cv2.Rodrigues(rvec)
        t_i = tvec
        recon.add_camera(i, R_i, t_i)

        # Triangulate new points between best_ref and the newly added camera i
        R_best, t_best = recon.camera_poses[best_ref]
        P_best = build_projection_matrix(K, R_best, t_best)
        P_i    = build_projection_matrix(K, R_i,    t_i)

        _, p_best, p_i2 = match_features(
            keypoints[best_ref], descriptors[best_ref],
            keypoints[i], descriptors[i],
            ratio=match_ratio,
        )
        if len(p_best) >= 5:
            pts3d_new = triangulate_points(P_best, P_i, p_best, p_i2)
            dm = filter_triangulated(pts3d_new, P_best, P_i)
            recon.add_points(pts3d_new, best_ref, i, p_best, p_i2, mask=dm)
            if verbose:
                print(f"  Added camera {i}. Triangulated {dm.sum()} new 3-D points.")
        else:
            if verbose:
                print(f"  Added camera {i} (no new triangulation).")

    return recon


# ---------------------------------------------------------------------------
# 8. Bundle Adjustment
# ---------------------------------------------------------------------------

def _project(points3d, camera_params, K):
    """Reproject 3-D points using rotation vector + translation."""
    rvec = camera_params[:3]
    tvec = camera_params[3:6]
    R, _ = cv2.Rodrigues(rvec)
    proj, _ = cv2.projectPoints(points3d, rvec, tvec, K, None)
    return proj.reshape(-1, 2)


def _ba_residuals(params, n_cams, n_pts, cam_indices, pt_indices, points2d, K):
    """Compute reprojection residuals for bundle adjustment (vectorized)."""
    cam_params = params[:n_cams * 6].reshape((n_cams, 6))
    pts3d = params[n_cams * 6:].reshape((n_pts, 3))

    cam_indices_arr = np.asarray(cam_indices)
    pt_indices_arr  = np.asarray(pt_indices)
    projected = np.empty_like(points2d)  # (N, 2)

    for ci in range(n_cams):
        mask = cam_indices_arr == ci
        if not mask.any():
            continue
        rvec = cam_params[ci, :3]
        tvec = cam_params[ci, 3:6]
        R, _ = cv2.Rodrigues(rvec)
        pts = pts3d[pt_indices_arr[mask]]       # (M, 3)
        pts_cam = pts @ R.T + tvec              # (M, 3)
        pts_h   = pts_cam @ K.T                 # (M, 3)
        projected[mask] = pts_h[:, :2] / pts_h[:, 2:3]

    return (projected - points2d).ravel()


def bundle_adjustment(
    recon: SfMReconstruction,
    max_nfev: int = 200,
    verbose: bool = True,
) -> SfMReconstruction:
    """Refine camera poses and 3-D points via Levenberg-Marquardt BA.

    Returns a new SfMReconstruction with refined parameters.
    """
    K = recon.K
    cam_idx_list = sorted(recon.camera_poses.keys())
    cam_id_to_local = {cid: i for i, cid in enumerate(cam_idx_list)}
    n_cams = len(cam_idx_list)

    track_ids = sorted(recon.points3d.keys())
    track_id_to_local = {tid: i for i, tid in enumerate(track_ids)}
    n_pts = len(track_ids)

    if n_cams == 0 or n_pts == 0:
        return recon

    # Initial parameter vector
    cam_params0 = []
    for cid in cam_idx_list:
        R, t = recon.camera_poses[cid]
        rvec, _ = cv2.Rodrigues(R)
        cam_params0.append(np.concatenate([rvec.ravel(), t.ravel()]))
    cam_params0 = np.concatenate(cam_params0)

    pts3d0 = np.array([recon.points3d[tid] for tid in track_ids]).ravel()
    x0 = np.concatenate([cam_params0, pts3d0])

    # Build observation lists
    cam_indices, pt_indices, pts2d_obs = [], [], []
    for tid, obs_list in recon.observations.items():
        if tid not in track_id_to_local:
            continue
        pi = track_id_to_local[tid]
        for cid, uv in obs_list:
            if cid not in cam_id_to_local:
                continue
            cam_indices.append(cam_id_to_local[cid])
            pt_indices.append(pi)
            pts2d_obs.append(uv)

    if len(pts2d_obs) == 0:
        return recon

    pts2d_obs = np.array(pts2d_obs)

    if verbose:
        print(f"Running BA: {n_cams} cameras, {n_pts} points, {len(pts2d_obs)} observations.")

    # Build sparse Jacobian sparsity pattern to avoid dense 21+ GiB allocation.
    # Each observation produces 2 residuals; each residual depends on 6 camera
    # params and 3 point params, so we mark those columns as non-zero.
    from scipy.sparse import lil_matrix
    n_obs = len(pts2d_obs)
    n_params = n_cams * 6 + n_pts * 3
    sparsity = lil_matrix((2 * n_obs, n_params), dtype=int)
    for i, (ci, pi) in enumerate(zip(cam_indices, pt_indices)):
        row = 2 * i
        sparsity[row:row + 2, ci * 6: ci * 6 + 6] = 1
        col_pt = n_cams * 6 + pi * 3
        sparsity[row:row + 2, col_pt: col_pt + 3] = 1

    result = least_squares(
        _ba_residuals,
        x0,
        jac_sparsity=sparsity,
        method="trf",
        args=(n_cams, n_pts, cam_indices, pt_indices, pts2d_obs, K),
        max_nfev=max_nfev,
        verbose=2 if verbose else 0,
    )

    # Unpack refined parameters
    x_opt = result.x
    cam_params_opt = x_opt[:n_cams * 6].reshape((n_cams, 6))
    pts3d_opt = x_opt[n_cams * 6:].reshape((n_pts, 3))

    refined = SfMReconstruction(K)
    for i, cid in enumerate(cam_idx_list):
        rvec = cam_params_opt[i, :3]
        tvec = cam_params_opt[i, 3:6]
        R, _ = cv2.Rodrigues(rvec)
        refined.add_camera(cid, R, tvec)

    for i, tid in enumerate(track_ids):
        refined.points3d[tid] = pts3d_opt[i]
        refined.observations[tid] = recon.observations[tid]
    refined._next_track = recon._next_track

    return refined


# ---------------------------------------------------------------------------
# 9. Reprojection Error
# ---------------------------------------------------------------------------

def compute_reprojection_error(recon: SfMReconstruction) -> float:
    """Mean reprojection error (pixels) over all observations (vectorized)."""
    K = recon.K

    pts3d_list, cid_list, uv_list = [], [], []
    for tid, obs_list in recon.observations.items():
        if tid not in recon.points3d:
            continue
        pt3d = recon.points3d[tid]
        for cid, uv in obs_list:
            if cid not in recon.camera_poses:
                continue
            pts3d_list.append(pt3d)
            cid_list.append(cid)
            uv_list.append(uv)

    if not pts3d_list:
        return float("nan")

    pts3d_arr = np.array(pts3d_list, dtype=np.float64)  # (N, 3)
    uv_arr    = np.array(uv_list,    dtype=np.float64)  # (N, 2)
    cid_arr   = np.array(cid_list)
    projected = np.empty_like(uv_arr)

    for cid, (R, t) in recon.camera_poses.items():
        mask = cid_arr == cid
        if not mask.any():
            continue
        pts = pts3d_arr[mask]
        pts_cam = pts @ R.T + t.ravel()         # (M, 3)
        pts_h   = pts_cam @ K.T                 # (M, 3)
        projected[mask] = pts_h[:, :2] / pts_h[:, 2:3]

    return float(np.mean(np.linalg.norm(projected - uv_arr, axis=1)))


# ---------------------------------------------------------------------------
# 9. Visualization Helpers
# ---------------------------------------------------------------------------

def show_keypoints(
    image: np.ndarray,
    keypoints: List[cv2.KeyPoint],
    title: str = "Keypoints",
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """Draw keypoints on an image."""
    drawn = cv2.drawKeypoints(
        image, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    drawn_rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(drawn_rgb)
    ax.set_title(f"{title}  ({len(keypoints)} kps)")
    ax.axis("off")
    return ax


def show_matches(
    img1: np.ndarray,
    kp1: List[cv2.KeyPoint],
    img2: np.ndarray,
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    mask: Optional[np.ndarray] = None,
    title: str = "Matches",
    max_draw: int = 100,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """Draw feature matches between two images."""
    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    if mask is not None:
        match_mask = [[1, 0] if m else [0, 0] for m in mask]
    else:
        match_mask = None

    drawn = cv2.drawMatchesKnn(
        img1, kp1, img2, kp2,
        [[m] for m in matches[:max_draw]],
        None,
        matchesMask=match_mask[:max_draw] if match_mask else None,
        **draw_params,
    )
    drawn_rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 6))
    ax.imshow(drawn_rgb)
    inlier_count = int(np.sum(mask)) if mask is not None else len(matches)
    ax.set_title(f"{title}  ({inlier_count} inliers shown)")
    ax.axis("off")
    return ax


def plot_epipolar_lines(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    F: np.ndarray,
    n_lines: int = 15,
    title: str = "Epipolar Lines",
) -> plt.Figure:
    """Draw epipolar lines on both images."""
    def draw_epilines(img, lines, pts):
        h, w = img.shape[:2]
        out = img.copy()
        for r, pt in zip(lines, pts):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / (r[1] + 1e-9)])
            x1, y1 = map(int, [w, -(r[2] + r[0] * w) / (r[1] + 1e-9)])
            cv2.line(out, (x0, y0), (x1, y1), color, 1)
            cv2.circle(out, tuple(map(int, pt.ravel())), 5, color, -1)
        return out

    idx = np.random.choice(len(pts1), min(n_lines, len(pts1)), replace=False)
    lines2 = cv2.computeCorrespondEpilines(pts1[idx].reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    lines1 = cv2.computeCorrespondEpilines(pts2[idx].reshape(-1, 1, 2), 2, F).reshape(-1, 3)

    out1 = draw_epilines(img1, lines1, pts1[idx])
    out2 = draw_epilines(img2, lines2, pts2[idx])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(cv2.cvtColor(out1, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"{title} – Image 1")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(out2, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"{title} – Image 2")
    axes[1].axis("off")
    fig.tight_layout()
    return fig


def create_camera_frustum(
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    scale: float = 0.1,
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> o3d.geometry.LineSet:
    """Create a camera frustum visualization for Open3D.
    
    Parameters
    ----------
    R : (3,3) rotation matrix (world to camera)
    t : (3,1) or (3,) translation vector
    K : (3,3) camera intrinsic matrix
    scale : frustum size
    color : RGB color tuple (0-1 range)
    
    Returns
    -------
    frustum : Open3D LineSet representing the camera frustum
    """
    # Camera center in world coordinates: C = -R^T @ t
    t_vec = t.ravel()
    cam_center = -R.T @ t_vec
    
    # Define image corners in normalized image coordinates
    w, h = 640, 480  # arbitrary image size for visualization
    corners_img = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1],
    ], dtype=np.float64)
    
    # Unproject to camera space
    K_inv = np.linalg.inv(K)
    corners_cam = (K_inv @ corners_img.T).T * scale
    
    # Transform to world space: x_world = R^T @ (x_cam - t)
    corners_world = (corners_cam - t_vec) @ R
    
    # Create frustum points: camera center + 4 corners
    points = np.vstack([cam_center.reshape(1, 3), corners_world])
    
    # Define lines connecting camera center to corners and corners to each other
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Center to corners
        [1, 2], [2, 3], [3, 4], [4, 1],  # Rectangle of image plane
    ]
    
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector([color for _ in lines])
    
    return frustum


def plot_3d_reconstruction(
    recon: SfMReconstruction,
    title: str = "SfM 3-D Reconstruction",
    point_size: float = 1.0,
    subsample: int = 5000,
    window_name: str = "SfM Reconstruction",
    images: Optional[List[np.ndarray]] = None,
) -> None:
    """Visualize 3-D points and camera poses using Open3D.
    
    Parameters
    ----------
    recon : SfMReconstruction object
    title : window title
    point_size : point cloud point size
    subsample : maximum number of points to display
    window_name : Open3D window name
    images : source images used for sampling point colours; if None, uniform blue is used
    """
    if images is not None:
        pts3d, colors_rgb = recon.get_points_and_colors(images)
    else:
        pts3d = recon.get_points_array()
        colors_rgb = None
    geometries = []
    
    if len(pts3d) > 0:
        # Remove outliers (simple percentile clip)
        mask = np.ones(len(pts3d), dtype=bool)
        for axis in range(3):
            low, high = np.percentile(pts3d[:, axis], [2, 98])
            mask &= (pts3d[:, axis] >= low) & (pts3d[:, axis] <= high)
        pts_filtered = pts3d[mask]
        colors_filtered = colors_rgb[mask] if colors_rgb is not None else None
        
        if len(pts_filtered) > subsample:
            idx = np.random.choice(len(pts_filtered), subsample, replace=False)
            pts_filtered = pts_filtered[idx]
            colors_filtered = colors_filtered[idx] if colors_filtered is not None else None
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_filtered)
        if colors_filtered is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors_filtered.astype(np.float64) / 255.0)
        else:
            pcd.paint_uniform_color([0.2, 0.5, 0.8])  # Steel blue fallback
        geometries.append(pcd)
    
    # Add camera frustums
    cam_ids = sorted(recon.camera_poses.keys())
    for cid in cam_ids:
        R, t = recon.camera_poses[cid]
        frustum = create_camera_frustum(R, t, recon.K, scale=0.2, color=(1.0, 0.0, 0.0))
        geometries.append(frustum)
    
    # Add camera trajectory line
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
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=1200,
        height=800,
        point_show_normal=False,
    )
    print(f"Displayed: {len(pts3d)} total points, {len(cam_ids)} cameras")


def plot_reprojection(
    recon: SfMReconstruction,
    images: List[np.ndarray],
    cam_idx: int = 0,
    max_points: int = 200,
    title: str = "Reprojection Check",
) -> plt.Figure:
    """Show observed vs reprojected points for one camera."""
    K = recon.K
    if cam_idx not in recon.camera_poses:
        raise ValueError(f"Camera {cam_idx} not in reconstruction.")
    R, t = recon.camera_poses[cam_idx]
    rvec, _ = cv2.Rodrigues(R)

    obs_pts, proj_pts = [], []
    for tid, obs_list in recon.observations.items():
        for cid, uv in obs_list:
            if cid == cam_idx and tid in recon.points3d:
                pt3d = recon.points3d[tid].reshape(1, 3).astype(np.float64)
                proj, _ = cv2.projectPoints(pt3d, rvec, t.astype(np.float64), K, None)
                obs_pts.append(uv)
                proj_pts.append(proj.ravel())

    obs_pts = np.array(obs_pts[:max_points])
    proj_pts = np.array(proj_pts[:max_points])

    fig, ax = plt.subplots(figsize=(10, 7))
    img_rgb = cv2.cvtColor(images[cam_idx], cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    if len(obs_pts) > 0:
        ax.scatter(obs_pts[:, 0], obs_pts[:, 1], s=8, c="lime", label="Observed", zorder=3)
        ax.scatter(proj_pts[:, 0], proj_pts[:, 1], s=8, marker="x", c="red",
                   label="Reprojected", zorder=4)
        for o, p in zip(obs_pts, proj_pts):
            ax.plot([o[0], p[0]], [o[1], p[1]], "y-", alpha=0.3, linewidth=0.7)
    mean_err = np.linalg.norm(obs_pts - proj_pts, axis=1).mean() if len(obs_pts) > 0 else float("nan")
    ax.set_title(f"{title}  (Camera {cam_idx}, mean err={mean_err:.2f}px)")
    ax.legend(loc="upper right")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_sfm_summary(
    recon: SfMReconstruction,
    n_matches_per_pair: Dict[Tuple[int, int], int],
    title: str = "SfM Pipeline Summary",
) -> plt.Figure:
    """Show a bar chart of inlier matches per pair and camera count."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Match count bar chart
    ax = axes[0]
    if n_matches_per_pair:
        labels = [f"{a}-{b}" for a, b in n_matches_per_pair.keys()]
        values = list(n_matches_per_pair.values())
        ax.bar(labels, values, color="steelblue")
        ax.set_xlabel("Image pair")
        ax.set_ylabel("Inlier matches")
        ax.set_title("Inlier matches per image pair")
        ax.tick_params(axis="x", rotation=45)

    # 3-D point count
    ax2 = axes[1]
    n_pts = len(recon.points3d)
    n_cams = len(recon.camera_poses)
    ax2.bar(["3-D points", "Cameras"], [n_pts, n_cams], color=["steelblue", "coral"])
    ax2.set_title("Reconstruction statistics")
    ax2.set_ylabel("Count")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig
