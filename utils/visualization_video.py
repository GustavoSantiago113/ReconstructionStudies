import os
import shutil
import tempfile
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import imageio.v2 as imageio


def _ensure_numpy_points(points: Union[np.ndarray, object]) -> np.ndarray:
	"""Convert supported point-cloud inputs to an (N,3) numpy array.

	Currently accepts numpy arrays or any object exposing `points` or `xyz` as
	a numpy-compatible array (e.g. open3d.geometry.PointCloud).
	"""
	if isinstance(points, np.ndarray):
		arr = points
	else:
		# duck-typing for Open3D PointCloud-like objects
		if hasattr(points, "points"):
			arr = np.asarray(points.points)
		elif hasattr(points, "xyz"):
			arr = np.asarray(points.xyz)
		else:
			raise TypeError("Unsupported point cloud type; provide a numpy array or Open3D PointCloud")

	if arr.ndim != 2 or arr.shape[1] < 3:
		raise ValueError("Points must be an (N,3) array")

	return arr[:, :3]


def generate_360_video(
	points: Union[np.ndarray, object],
	output_path: str,
	n_frames: int = 360,
	elevation: float = 20.0,
	align_pca: bool = True,
	center: Optional[Union[tuple, list, np.ndarray]] = None,
	temp_dir: Optional[str] = None,
	fps: int = 30,
	figsize: tuple = (8, 8),
	bg_color: str = "white",
	point_size: float = 1.0,
	cmap: Optional[Union[str, np.ndarray]] = None,
	remove_images: bool = True,
):
	"""Render a 360-degree rotating video of a point cloud.

	- `points`: (N,3) numpy array or Open3D PointCloud-like object.
	- `output_path`: path to write the final video (e.g. 'out.mp4').
	- `n_frames`: number of frames around full rotation.
	- `elevation`: elevation angle for the camera.
	- `center`: center point to orbit around (defaults to mean).
	- `temp_dir`: directory to save intermediate PNGs (auto-created if None).
	- `fps`: frames per second of output video.
	- `figsize`: matplotlib figure size.
	- `bg_color`: background color for frames.
	- `point_size`: marker size for scatter.
	- `cmap`: either a color string or per-point color array.
	- `remove_images`: delete intermediate images after video creation.
	"""
	# allow passing a tuple (points, colors) or a pointcloud-like object
	if isinstance(points, tuple) and len(points) >= 1:
		pts = _ensure_numpy_points(points[0])
		colors = None
		if len(points) > 1:
			colors = np.asarray(points[1])
	else:
		pts = _ensure_numpy_points(points)
		colors = None

	print(f"[video] Starting generation: output={output_path}, frames={n_frames}, fps={fps}, bg={bg_color}")

	# PCA alignment (optional): rotate points into principal axes
	if align_pca:
		pca_mean = pts.mean(axis=0)
		pts_centered = pts - pca_mean
		# SVD for PCA
		U, S, Vt = np.linalg.svd(pts_centered, full_matrices=False)
		# project points into principal component basis
		pts = pts_centered.dot(Vt.T)
		print(f"[video] Applied PCA alignment (mean shifted).")
		# if colors exist, they don't need rotation but remain aligned to pts
		center = np.zeros(3)
	else:
		if center is None:
			center = pts.mean(axis=0)
		center = np.asarray(center)
		# translate to center
		pts = pts - center

	# setup temp directory
	cleanup_temp_dir = False
	if temp_dir is None:
		temp_dir = tempfile.mkdtemp(prefix="pc_video_")
		cleanup_temp_dir = True
	else:
		os.makedirs(temp_dir, exist_ok=True)

	# prepare plotting
	rcParams["figure.dpi"] = 100
	fig = plt.figure(figsize=figsize)
	fig.patch.set_facecolor(bg_color)
	ax = fig.add_subplot(111, projection="3d")
	ax.set_facecolor(bg_color)

	xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]

	# colors: prefer explicit colors variable (from file), else use cmap
	if colors is None:
		if cmap is None:
			# dark points on white background
			colors = "black" if bg_color.lower() in ("white", "#ffffff") else "white"
		elif isinstance(cmap, str):
			colors = cmap
		else:
			colors = np.asarray(cmap)
	else:
		# normalize colors if needed (0-255 -> 0-1)
		col_arr = np.asarray(colors)
		if col_arr.dtype == np.uint8 or col_arr.max() > 1.5:
			col_arr = col_arr.astype(np.float32) / 255.0
		# ensure shape (N,3) or (N,4)
		colors = col_arr

	# scatter once, we'll only change the view
	scatter = ax.scatter(xs, ys, zs, c=colors, s=point_size, linewidths=0, depthshade=False)

	# set equal axis limits
	max_range = np.array([xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()]).max() / 2.0
	mid_x = (xs.max() + xs.min()) * 0.5
	mid_y = (ys.max() + ys.min()) * 0.5
	mid_z = (zs.max() + zs.min()) * 0.5
	ax.set_xlim(mid_x - max_range, mid_x + max_range)
	ax.set_ylim(mid_y - max_range, mid_y + max_range)
	ax.set_zlim(mid_z - max_range, mid_z + max_range)

	# remove axes for a clean look
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_zticks([])
	ax.grid(False)
	ax.set_axis_off()

	# render frames
	filenames = []
	try:
		# choose a print interval so we don't spam the console
		print_interval = max(1, n_frames // 10)
		for i in range(n_frames):
			azim = 360.0 * float(i) / float(n_frames)
			ax.view_init(elev=elevation, azim=azim)

			fname = os.path.join(temp_dir, f"frame_{i:04d}.png")
			# tight layout + transparent backgrounds can crop the point cloud; use bbox_inches='tight' and facecolor
			fig.savefig(fname, dpi=rcParams["figure.dpi"], facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
			filenames.append(fname)
			if (i + 1) % print_interval == 0 or i == n_frames - 1:
				print(f"[video] Rendered frame {i+1}/{n_frames}")

		# write video using imageio (ffmpeg). Explicitly request ffmpeg
		# to avoid other writers (e.g. tifffile) receiving unexpected kwargs.
		print(f"[video] Writing video to {output_path} (ffmpeg)")
		try:
			with imageio.get_writer(output_path, format="ffmpeg", fps=fps, codec="libx264", quality=8, ffmpeg_params=["-pix_fmt", "yuv420p"]) as writer:
				for fname in filenames:
					img = imageio.imread(fname)
					writer.append_data(img)
		except Exception as e:
			raise RuntimeError(
				"Failed to write video with ffmpeg. Ensure 'imageio-ffmpeg' is installed and ffmpeg is available on PATH."
			) from e
		print(f"[video] Video saved: {output_path}")

	finally:
		if remove_images:
			print(f"[video] Cleaning up {len(filenames)} temporary images in {temp_dir}")
			for fname in filenames:
				try:
					os.remove(fname)
				except Exception:
					pass
			if cleanup_temp_dir:
				try:
					shutil.rmtree(temp_dir)
					print(f"[video] Removed temporary directory {temp_dir}")
				except Exception:
					pass


def generate_360_video_from_file(
	input_path: str,
	output_path: str,
	n_frames: int = 360,
	elevation: float = 20.0,
	**kwargs,
):
	"""Load a point cloud file (PLY/XYZ/NPY) and generate a 360° video.

	Tries to use `open3d` if available for PLY/PLY-like formats. If given a
	`.npy` file it will load a numpy array directly.
	"""
	colors = None
	if input_path.lower().endswith(".npy"):
		arr = np.load(input_path)
		if arr.ndim == 2 and arr.shape[1] >= 3:
			pts = arr[:, :3]
			if arr.shape[1] >= 6:
				colors = arr[:, 3:6]
		else:
			raise RuntimeError(f"Unsupported .npy shape for point cloud: {arr.shape}")
	else:
		try:
			import open3d as o3d

			p = o3d.io.read_point_cloud(input_path)
			pts = np.asarray(p.points)
			if hasattr(p, 'colors') and len(p.colors) == len(p.points):
				colors = np.asarray(p.colors)
		except Exception:
			# fallback: try to load ASCII XYZ or XYZRGB
			try:
				arr = np.loadtxt(input_path)
				if arr.ndim == 2 and arr.shape[1] >= 3:
					pts = arr[:, :3]
					if arr.shape[1] >= 6:
						colors = arr[:, 3:6]
				else:
					raise RuntimeError(f"Unsupported ascii point cloud shape: {arr.shape}")
			except Exception as e:
				raise RuntimeError(f"Could not read point cloud from {input_path}: {e}")

	if colors is not None:
		generate_360_video((pts, colors), output_path, n_frames=n_frames, elevation=elevation, **kwargs)
	else:
		generate_360_video(pts, output_path, n_frames=n_frames, elevation=elevation, **kwargs)


if __name__ == "__main__":
	# tiny example: generates a rotating sphere point cloud if run standalone
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("--out", "-o", default="pc360.mp4")
	parser.add_argument("--in", "-i", dest="input", default=None,
			help="Input point cloud file (PLY, NPY, or ASCII XYZ). If provided, generates video from this file.")
	parser.add_argument("--frames", "-n", type=int, default=180)
	parser.add_argument("--fps", type=int, default=30)
	parser.add_argument("--no-pca", dest="no_pca", action="store_true", help="Disable PCA alignment (default: enabled)")
	args = parser.parse_args()

	generate_360_video_from_file(args.input, args.out, n_frames=args.frames, fps=args.fps, align_pca=not args.no_pca)