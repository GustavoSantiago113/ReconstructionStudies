import os
import numpy as np

def select_equally_distributed_images(image_folder, n):
    """
    Select n equally distributed stop_{number} images from the folder.
    Returns a list of file paths.
    """
    # Get all stop_{number} images, sorted by number
    files = [f for f in os.listdir(image_folder) if f.startswith('stop_') and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    # Extract numbers and sort
    numbered = sorted([(int(f.split('_')[1][:2]), f) for f in files])
    if not numbered:
        return []
    indices = np.linspace(0, len(numbered)-1, n, dtype=int)
    selected = [os.path.join(image_folder, numbered[i][1]) for i in indices]
    return selected