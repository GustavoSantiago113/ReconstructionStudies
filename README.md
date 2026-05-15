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

3. Get the data. 

- Go to https://cseweb.ucsd.edu//~viscomp/projects/LF/papers/ECCV20/nerf/ and download the nerf_example_data.zip
- Unzip the downloaded file and click on it.
- Go to `nerf_example_data\nerf_synthetic\lego\`.
- Copy the files from `train\` and paste into the folder [data/](data/).

4. Open the [notebooks/](notebooks/):

Run the notebooks to reproduce experiments and view final outputs in [results/](results/)

## Results

Below is the results table regarding processing time of each method.

| Method | Images Used | Reconstruction Time | Requires GPU? |
| ------ | ------------------- | ----------- | ------------- |
| COLMAP | 100 | 40 min | For Dense Reconstruction |
| Meshroom | 100 | 10min | For Dense Reconstruction |
| NeRF | 100 | 3:38 h | Yes |
| Dust3r | 10 | ~ 1min | No, but makes it faster |
| VGGT | 10 | ~ 2min | No, but makes it faster |
| DepthAnythingv3 | 10 | ~ 3min | No, but makes it faster |

## Demonstrations

See the 360 of the output reconstructions for each method below:

- COLMAP:


## License & citation

- See LICENSE at repo root for repository license.
- All the models used are from papers presented and cited in `docs/methodsDescription.md`.

## DIY

