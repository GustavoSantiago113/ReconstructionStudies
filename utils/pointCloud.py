# Clean and visualize a point cloud using Open3D
import open3d as o3d
import numpy as np

def voxel_farthest_point_sampling(pcd, voxel_size=0.05, num_samples=None):
    """
    Voxel-based Farthest Point Sampling (VFPS) for point cloud downsampling.
    
    Args:
        pcd: Open3D PointCloud object
        voxel_size: Size of voxel grid for initial downsampling
        num_samples: Target number of points after FPS (if None, uses voxel downsampling result)
    
    Returns:
        Cleaned and downsampled Open3D PointCloud
    """
    # Step 1: Voxel downsampling for initial reduction
    pcd_voxel = pcd.voxel_down_sample(voxel_size)
    
    # Step 2: Apply farthest point sampling
    points = np.asarray(pcd_voxel.points)
    n_points = len(points)
    
    if num_samples is None or num_samples >= n_points:
        return pcd_voxel
    
    # FPS algorithm
    selected_indices = np.zeros(num_samples, dtype=np.int32)
    distances = np.full(n_points, np.inf)
    
    # Start with a random point
    selected_indices[0] = np.random.randint(0, n_points)
    
    for i in range(1, num_samples):
        # Update distances to the last selected point
        last_selected = selected_indices[i-1]
        last_point = points[last_selected]
        new_distances = np.linalg.norm(points - last_point, axis=1)
        distances = np.minimum(distances, new_distances)
        
        # Select the farthest point
        selected_indices[i] = np.argmax(distances)
    
    # Create new point cloud with sampled points
    pcd_sampled = pcd_voxel.select_by_index(selected_indices.tolist())
    
    return pcd_sampled

def clean_point_cloud(pcd_or_path, method='statistical', nb_neighbors=20, std_ratio=2.0, radius=0.05, min_points=16, voxel_size=None, fps_samples=None):
    """
    Clean a point cloud using Open3D outlier removal and optional downsampling.
    - pcd_or_path: Open3D PointCloud or path to PLY/PCD file
    - method: 'statistical', 'radius', or 'vfps' (voxel-based farthest point sampling)
    - voxel_size: for VFPS, size of voxel grid
    - fps_samples: for VFPS, target number of points after sampling
    - returns cleaned Open3D PointCloud
    """
    if isinstance(pcd_or_path, str):
        pcd = o3d.io.read_point_cloud(pcd_or_path)
    else:
        pcd = pcd_or_path

    if method == 'vfps':
        # Use voxel-based farthest point sampling
        if voxel_size is None:
            voxel_size = 0.05  # default voxel size
        pcd_clean = voxel_farthest_point_sampling(pcd, voxel_size=voxel_size, num_samples=fps_samples)
    elif voxel_size is not None and method != 'vfps':
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
    elif method == 'vfps':
        # Already handled above
        pass
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