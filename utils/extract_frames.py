"""Extract frames from video files.

Usage:
	python utils/extract_frames.py --video PATH_TO_VIDEO [--outdir frames] [--rate 5]

Defaults:
	outdir: ./frames
	rate: 5  # frames per second to extract

The script attempts to read video FPS and duration. If FPS is available it will sample
frames at the specified rate (every 1/rate seconds). If FPS is not available it will
fall back to reading frames and use timestamps reported by OpenCV.
"""

from pathlib import Path
import argparse
import cv2
import numpy as np
import math
import sys


def extract_frames(video_path, out_dir='frames', rate=5, verbose=True):
	"""Extract frames from a video at `rate` frames per second.

	Args:
		video_path (str or Path): path to video file
		out_dir (str or Path): directory to save frames
		rate (int): frames per second to extract
		verbose (bool): print progress
	Returns:
		int: number of frames extracted
	"""
	video_path = Path(video_path)
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise RuntimeError(f"Failed to open video: {video_path}")

	# Try to get FPS and frame count
	fps = cap.get(cv2.CAP_PROP_FPS)
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
	duration = None
	if fps and frame_count:
		duration = frame_count / fps
		if verbose:
			print(f"Video FPS={fps:.2f}, frames={frame_count}, duration={duration:.2f}s")

	# Compute timestamps to extract: every 1/rate seconds
	timestamps = None
	if duration is not None and duration > 0:
		n = max(0, int(math.floor(duration * rate)))
		timestamps = np.arange(0, n) / float(rate)
	else:
		# Unknown duration: we'll sample based on POS_MSEC while reading
		timestamps = None

	saved = 0
	if timestamps is not None and len(timestamps) > 0:
		# Convert timestamps to frame indices if fps is known
		if fps and fps > 0:
			frame_indices = np.unique(np.round(timestamps * fps).astype(int))
			frame_indices = frame_indices[frame_indices < frame_count]
			if verbose:
				print(f"Extracting {len(frame_indices)} frames at {rate} fps")

			for idx in frame_indices:
				cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
				ret, frame = cap.read()
				if not ret:
					continue
				out_path = out_dir / f"frame_{saved:06d}.png"
				cv2.imwrite(str(out_path), frame)
				saved += 1
		else:
			# fallback: use timestamps and set POS_MSEC
			for t in timestamps:
				cap.set(cv2.CAP_PROP_POS_MSEC, float(t * 1000.0))
				ret, frame = cap.read()
				if not ret:
					continue
				out_path = out_dir / f"frame_{saved:06d}.png"
				cv2.imwrite(str(out_path), frame)
				saved += 1
	else:
		# Unknown duration/fps — read frame-by-frame and sample using reported timestamp
		if verbose:
			print("Unknown duration/FPS — sampling using frame timestamps")
		next_t = 0.0
		while True:
			ret, frame = cap.read()
			if not ret:
				break
			# POS_MSEC is milliseconds
			t_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
			t_sec = t_msec / 1000.0
			if t_sec + 1e-6 >= next_t:
				out_path = out_dir / f"frame_{saved:06d}.png"
				cv2.imwrite(str(out_path), frame)
				saved += 1
				next_t += 1.0 / float(rate)

	cap.release()
	if verbose:
		print(f"Saved {saved} frames to: {out_dir}")
	return saved


def main():
	parser = argparse.ArgumentParser(description="Extract frames from video at N fps")
	parser.add_argument("--video", "-v", required=True, help="Path to input video file")
	parser.add_argument("--outdir", "-o", default="frames", help="Output directory for frames")
	parser.add_argument("--rate", "-r", type=float, default=5.0, help="Frames per second to extract")
	args = parser.parse_args()

	try:
		n = extract_frames(args.video, args.outdir, rate=args.rate, verbose=True)
		print(f"Done — extracted {n} frames.")
	except Exception as e:
		print(f"Error: {e}")
		sys.exit(1)


if __name__ == '__main__':
	main()

