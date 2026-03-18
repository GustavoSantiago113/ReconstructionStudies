import numpy as np
import open3d as o3d


def poisson_reconstruction_from_point_cloud(
    pcd, output_mesh_file, depth=15, width=0, scale=1.1, linear_fit=False
):
    """
    Perform Poisson surface reconstruction from a point cloud with advanced processing.
    Includes mesh cleaning, smoothing, simplification, and color transfer.

    Parameters:
    -----------
    pcd : o3d.geometry.PointCloud
        Input point cloud
    output_mesh_file : str
        Path to save the output mesh file (typically .ply or .obj)
    depth : int
        Maximum depth of the octree used for reconstruction (higher=more detail but slower)
    width : int
        Specifies the target width of the finest level of the octree (0=automatic)
    scale : float
        Ratio between the diameter of the cube used for reconstruction and the diameter of the input point cloud
    linear_fit : bool
        If true, use linear interpolation to fit the implicit function
    """

    # Check if normals exist, estimate them if they don't
    if not pcd.has_normals():
        print("Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

    print("Running Poisson surface reconstruction...")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
        )

    # Filter out low-density vertices
    print("Filtering low-density vertices...")
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # Clean topology: remove degenerate/duplicated/non-manifold elements
    print("Cleaning mesh topology...")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Laplacian smoothing (removes local artifacts)
    print("Applying Taubin smoothing...")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=10)

    # Clean up after smoothing
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()

    # Simplify to reduce vertices (30% of original triangle count)
    print("Simplifying mesh...")
    target_triangle_count = max(1000, int(len(mesh.triangles) * 0.3))
    mesh_simplified = mesh.simplify_quadric_decimation(target_triangle_count)

    # Recompute normals after simplification
    mesh_simplified.compute_vertex_normals()

    # Transfer colors from point cloud to mesh
    print("Transferring colors to mesh...")
    if pcd.has_colors():
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        mesh_colors = []
        default_color = [0.5, 0.5, 0.5]  # Fallback color (mid gray)
        
        for v in mesh_simplified.vertices:
            k, idx, _ = pcd_tree.search_knn_vector_3d(v, 1)
            if k > 0:
                mesh_colors.append(pcd.colors[idx[0]])
            else:
                mesh_colors.append(default_color)
        
        mesh_simplified.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
    else:
        print("Point cloud has no colors; mesh will not be colored.")

    # Save the mesh
    print(f"Saving reconstructed mesh to {output_mesh_file}...")
    o3d.io.write_triangle_mesh(output_mesh_file, mesh_simplified)

    return mesh_simplified