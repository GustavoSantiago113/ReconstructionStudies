# Reconstruction Studies

This repository is a compilation of AI-Based 3D reconstruction methods and tools for processing, cleaning, and visualizing point clouds and meshes. It integrates various libraries and frameworks to facilitate research and development in 3D reconstruction workflows.

## Project Structure

- **Dust3r**: A submodule for 3D reconstruction and visualization.
- **VGG-T**: A framework for 3D reconstruction and texture mapping.
- **Depth-Anything-3 (DA3)**: A foundation model for spatially consistent geometry, supporting monocular/multi-view depth and pose estimation.
- **Notebooks**: Jupyter notebooks for implementing the frameworks on images, experimenting with point cloud cleaning, meshing, and texturing and segmenting images for model input.
- **Utils**: Utility scripts for preprocessing, segmentation, and reconstruction pipelines.
- **Models**: Pretrained models for segmentation.

## Installation

Clone locally:

```bash
git clone --recursive https://github.com/GustavoSantiago113/ReconstructionStudies
cd ReconstructionStudies
pip install -r requirements.txt
```

## Usage

To use the frameworks, I have made notebooks for each one. [dust3r](./notebooks/dust3r.ipynb), [depthAnythingv3](./notebooks/depthAnything3.ipynb), and [vggt](./notebooks/vggt.ipynb). You can test on the images in this repository or use your own. On the second option, just remember to change the image paths.

## Segmentation and Visualization Improvements

### Segmentation

The `notebooks/improveVisuals.ipynb` notebook demonstrates the use of segmentation models to enhance point cloud cleaning and reconstruction workflows. It was applied to create a segmentation model for the miniatures:

- **U-NET**: Used for generating masks to isolate relevant regions in images to be re-projected to point clouds. Found out to be the best model among others.

### Visualization Improvements

The `notebooks/improveVisuals.ipynb` notebook also includes methods for:

- **Point Cloud Cleaning tested methods**:
  - Radius-based cleaning to remove sparse outliers.
  - Statistical outlier removal for noise reduction.
  - Voxel-based Farthest Point Sampling (VFPS) to reduce point cloud size.

- **Mesh Generation**:
  - Poisson surface reconstruction for high-quality meshes.
  - Topology cleaning to remove degenerate and non-manifold elements.
  - Laplacian smoothing to reduce local artifacts.
  - Mesh simplification to optimize vertex count while preserving detail.

These techniques ensure that the reconstructed models are both visually appealing and computationally efficient.

## Depth Anything v3

**Key Takeaway:** For both datasets, using **4 images with segmentation** for both crops and miniatures provided the best balance between reconstruction quality and processing time.

## Dust3r

**Key Takeaway:** Dust3r performs well with **4 segmented images** for miniatures and **10 segmented images** for crops, providing a good balance between quality and processing time.

## VGGT

**Key Takeaway:** VGGT achieves the best quality with **10 non segmented images**, particularly for Miniatures, and **8 non segmented images** for crops while maintaining reasonable processing times.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the repository.

## License

This repository includes submodules and dependencies with their own licenses. Please refer to the respective `LICENSE` files for details.