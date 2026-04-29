import glob
import cv2
import os
import numpy as np

# Parameters
CHECKERBOARD = (22, 16) # inner-corner pattern (cols, rows)
SQUARE_SIZE = 10 # mm –– CHANGE THIS VALUE LATER
SHOW = True # draw detections

# Prepare the world points
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE # scale to real-world units

objpoints, imgpoints = [], [] # 3-D, 2-D correspondences
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Loop through all calibration pictures
for f in sorted(glob.glob("./data/calibration/*.png")):
    img = cv2.imread(f)
    if img is None:
        print(f"{f}: could not read file")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Stretch contrast, helps on dull prints:
    gray_eq = cv2.equalizeHist(gray)

    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FAST_CHECK)

    ret, corners = cv2.findChessboardCorners(gray_eq, CHECKERBOARD, flags)

    # Fallback to a more robust SB algorithm
    if not ret and hasattr(cv2, "findChessboardCornersSB"):
        ret, corners = cv2.findChessboardCornersSB(
            gray_eq, CHECKERBOARD,
            cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY |
            cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)

        print(f"{os.path.basename(f):15s}  ✓  {len(corners)} corners")
        if SHOW:
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners, True)
            cv2.imshow("detected", img); cv2.waitKey(250)
    else:
        print(f"{os.path.basename(f):15s}  ✗  not detected")

cv2.destroyAllWindows()
print(f"\nDetected corners in {len(objpoints) } / {len(imgpoints)+len(objpoints)} images")

# Calibrate
if len(objpoints) < 3:
    raise RuntimeError("Need at least 3 good views for a usable calibration")

rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n=== Calibration result ===")
print(f"RMS reprojection error: {rms:.4f} px")
print("Camera matrix (K):\n", K)
print("Distortion coeffs  :", dist.ravel())

# Per-image reprojection error
total_err = 0
for i in range(len(objpoints)):
    imgpts2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    err = cv2.norm(imgpoints[i], imgpts2, cv2.NORM_L2) / len(imgpts2)
    total_err += err
    print(f"Frame {i:02d}: error = {err:.4f} px")
print(f"Mean reprojection error: {total_err/len(objpoints):.4f} px")