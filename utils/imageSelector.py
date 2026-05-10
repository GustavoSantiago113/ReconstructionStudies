import os
import numpy as np

def select_equally_distributed_images(image_folder, n):
    """
    Select n equally distributed images from the folder.
    Returns a list of file paths.
    """
    # Get all images, sorted
    files = sorted([f for f in os.listdir(image_folder)])
    if not files:
        return []
    indices = np.linspace(0, len(files)-1, n, dtype=int)
    selected = [os.path.join(image_folder, files[i]) for i in indices]
    return selected