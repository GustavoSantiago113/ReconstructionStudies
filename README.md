<div align="center">

# Reconstruction Studies

**Gustavo Nocera Santiago, 2026**

</div>

---

A compact workspace for experiments in 3D reconstruction. This repo collects utilities, example notebooks, data and model artifacts used for reconstruction experiments.

## Contents

- `notebooks` — Jupyter notebooks with experiments and pipelines  
- `utils` — helper scripts (frame extraction, nerf models, image selector, meshing tools, camera calibration).  
- `data` — images and the source video(s) used for the experiments.
- `results` — output reconstructions, point clouds and visualizations.
- `docs` - studying notes regarding each reconstruction method tested here.

## Quick start

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
pip install jupyterlab opencv-python numpy
```

3. Extract frames from a video (example script included in [utils/](utils/))
```bash
python utils/extract_frames.py --video path/to/video.mp4 --outdir data/frames --rate 5
```

This saves sampled frames to data/frames (default rate = 5 FPS).

4. Open the [notebooks/](notebooks/):

Run the notebooks to reproduce experiments and view intermediate outputs in [results/](results/)

## Examples and demos

- extract_frames.py — quick frame extraction from video to use the images as methods' inputs.
- See [notebooks/](notebooks/) for step-by-step experiments and visualization cells.

## Results

Below is the results table regarding processing time of each method.

| Method | Images Used | Reconstruction Time | Requires GPU? |
| ------ | ------------------- | ----------- | ------------- |
| COLMAP | 112 | ~ 1h | For Dense Reconstruction |
| Meshroom | 112 | 23min | For Dense Reconstruction |
| NeRF | 112 | ~ 9h | Yes |
| Dust3r | 10 | ~ 1min | No, but makes it faster |
| VGGT | 10 | ~ 2min | No, but makes it faster |
| DepthAnythingv3 | 10 | ~ 3min | No, but makes it faster |
| RGBD Turntable | 36 | | No, but makes it faster |

## Demonstrations

See the 360 of the output reconstructions for each method below:

- COLMAP Apple:

<video src="./media/COLMAP_Apple.mp4" controls preload></video>

- COLMAP Miniature:

<video src="./media/COLMAP_Miniature.mp4" controls preload></video>


## License & citation

- See LICENSE at repo root for repository license.
- All the models used are from papers presented and cited in `docs/methodsDescription.md`.