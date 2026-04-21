<center>

# Reconstruction Studies

**Gustavo Nocera Santiago, 2026**

</center>

---

A compact workspace for experiments in 3D reconstruction and depth-from-video. This repo collects utilities, example notebooks, data and model artifacts used for reconstruction experiments.

## Contents

- `notebooks` — Jupyter notebooks with experiments and pipelines  
- `utils` — helper scripts (frame extraction, colmap helpers, point-cloud and mesh tools)  
- `data` — example images and the source video(s) used for experiments  
- `models` — saved model checkpoints and model artifacts (if present)  
- `results` — output reconstructions, point clouds and visualizations
- `docs` - studying notes regarding each reconstruction method presented here

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

```bash
jupyter lab notebooks/
```

Run the notebooks to reproduce experiments and view intermediate outputs in [results/](results/)

## Examples and demos

- See [notebooks/](notebooks/) for step-by-step experiments and visualization cells.
- extract_frames.py — quick frame extraction from video (useful pre-processing).
- colmap.py, pointCloud.py, and other scripts contain small helpers for reconstruction pipelines.

## Demonstrations

See the 360 of the output reconstructions for each method below:

## Results

Below is the results table regarding processing time and reconstruction quality of each method.


## License & citation

- See LICENSE at repo root for repository license.
- All the models used are from papers presented in `docs/methodsDescription.md`.