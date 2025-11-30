# Clean and visualize a point cloud using Open3D
import open3d as o3d
import numpy as np

def clean_point_cloud(pcd_or_path, method='statistical', nb_neighbors=20, std_ratio=2.0, radius=0.05, min_points=16, voxel_size=None):
    """
    Clean a point cloud using Open3D outlier removal and optional downsampling.
    - pcd_or_path: Open3D PointCloud or path to PLY/PCD file
    - method: 'statistical' or 'radius'
    - returns cleaned Open3D PointCloud
    """
    if isinstance(pcd_or_path, str):
        pcd = o3d.io.read_point_cloud(pcd_or_path)
    else:
        pcd = pcd_or_path

    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size)

    if method == 'statistical':
        pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        # remove_statistical_outlier returns (clipped_cloud, indices) in newer open3d; handle both styles
        if isinstance(pcd_clean, tuple):
            # older API returned (pcd_clean, ind)
            pcd_clean = pcd.select_by_index(pcd_clean[1])
    elif method == 'radius':
        pcd_clean, ind = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)
        if isinstance(pcd_clean, tuple):
            pcd_clean = pcd.select_by_index(pcd_clean[1])
    else:
        raise ValueError('Unknown method: ' + str(method))

    # Ensure normals exist for some visualizers (optional)
    try:
        if not pcd_clean.has_normals():
            pcd_clean.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    except Exception:
        pass

    return pcd_clean

def visualize_point_cloud(pcd_or_path, window_name='Point Cloud'):
    if isinstance(pcd_or_path, str):
        pcd = o3d.io.read_point_cloud(pcd_or_path)
    else:
        pcd = pcd_or_path
    # Use the simple blocking visualizer; interactive GUI will open
    o3d.visualization.draw_geometries([pcd], window_name=window_name)