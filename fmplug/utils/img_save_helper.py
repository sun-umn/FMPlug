import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os

def save_png_cv2(array, path):
    # If array is CHW, convert to HWC
    if array.ndim == 3 and array.shape[0] in [1, 3]:
        array = np.transpose(array, (1, 2, 0))

    # Clip and convert to uint8 if needed
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 1)
        array = (array * 255).round().astype(np.uint8)

    # If 3-channel RGB, convert to BGR for OpenCV
    if array.ndim == 3 and array.shape[2] == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    cv2.imwrite(path, array)

def visualize_image(ref: np.array, y_n: np.array, output: np.array, save_file_name: str) -> None:
    # Create a figure with 1 row, 3 columns
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 10))

    # Ground truth
    ax1.imshow(ref)
    ax1.set_title("Ground Truth Image")
    ax1.axis("off")

    # Display the corrupted image
    ax2.imshow(y_n)
    ax2.set_title("Corrupted Image")
    ax2.axis("off")

    # Reconstructed image
    ax3.imshow(output)
    ax3.set_title("Reconstructed Image (FM)")
    ax3.axis("off")

    # Pixel-wise absolute difference (grayscale)
    diff = np.abs(ref - output).astype(float).mean(axis=-1)
    ax4.imshow(diff, cmap="hot")
    ax4.set_title("Pixel Difference (FM)")
    ax4.axis("off")

    # Save the figure
    plt.tight_layout()
    fig.savefig(
        os.path.join(save_file_name),
        bbox_inches="tight",
    )
    plt.close()