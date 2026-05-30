<div align="center">

# 3D Reconstruction Studies

</div>

---

- [3D Reconstruction Studies](#3d-reconstruction-studies)
- [Contents](#contents)
- [Quick start - Reproducing the Results](#quick-start---reproducing-the-results)
- [Results](#results)
- [Reconstruction Visualization](#reconstruction-visualization)
- [License \& citation](#license--citation)
- [DIY](#diy)

---

A compact workspace for to learn 3D reconstruction. This repo collects utilities, example notebooks, data and model artifacts used for reconstruction experiments. It comprises the 3D reconstruction pipelines:

- COLMAP
- DepthAntythingv3
- dust3r
- Meshroom
- NeRF
- VGGT

# Contents

- `notebooks` — Jupyter notebooks with experiments and pipelines  
- `utils` — helper scripts (frame extraction, nerf models, image selector, meshing tools, camera calibration).  
- `data` — images and the source video(s) used for the experiments.
- `results` — output reconstructions, point clouds and visualizations.
- `docs` - studying notes regarding each reconstruction method tested here.

# Quick start - Reproducing the Results

1. Create a Python environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies (project-wide or per-script as needed). Example (install common tools):

```bash
pip install requirements.txt
```

3. Get the data. 

- Go to https://cseweb.ucsd.edu//~viscomp/projects/LF/papers/ECCV20/nerf/ and download the nerf_example_data.zip
- Unzip the downloaded file and click on it.
- Go to `nerf_example_data\nerf_synthetic\lego\`.
- Copy the files from `train\` and paste into the folder [data/](data/).

4. Open the [notebooks/](notebooks/):

Run the notebooks to reproduce experiments and view final outputs in [results/](results/)

# Results

Below is the results table regarding processing time of each method.

| Method | Images Used | Reconstruction Time | Requires GPU? |
| ------ | ------------------- | ----------- | ------------- |
| COLMAP | 100 | 40 min | For Dense Reconstruction |
| Meshroom | 100 | 10min | For Dense Reconstruction |
| NeRF | 100 | 3:38 h | Yes |
| Dust3r | 10 | ~ 1min | No, but makes it faster |
| VGGT | 10 | ~ 2min | No, but makes it faster |
| DepthAnythingv3 | 10 | ~ 3min | No, but makes it faster |

# Reconstruction Visualization

See the 360 view video of the output reconstructions for each method below:

- [COLMAP](results/colmap_results/fused.ply): [video]("media/colmap.mp4")

- [Meshroom](results/meshroom/point_cloud.ply): [video]("media/meshroom.mp4")

- [NeRF](results/nerf/nerf_pointcloud.ply): [video]("media/nerf.mp4")

- [Dust3r](results/dust3r/reconstruction.ply): [video]("media/dust3r.mp4")

- [VGGT](results/vggt/Reconstruction.ply): [video]("media/vggt.mp4")

- [Depth Anything v3](results/da3/Reconstruction.ply): [video]("media/da3.mp4")


# License & citation

- See LICENSE at repo root for repository license.
- All the pipelines, models and methods used has a more detailed explanation presented and the papers that originated them cited in [methodsDescription.md](docs/methodsDescription.md).

# DIY

1. Get multiple view images from an object or scene.
   1. You can record a video and use [extract_frames.py](utils/extract_frames.py) script to extract the frames.
2. Save the data in `data/` folder.
3. Use the scripts from the `notebooks/` again to generate the 3D reconstructions.
4. Generate the 360 view video from the reconstruction using the script [visualization_video.py](utils/visualization_video.py)
