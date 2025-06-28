import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from src.utilities import image_utils, path_utils


def plot_comparison_matrix(
    original_images: list[torch.Tensor],
    original_garments: list[torch.Tensor],
    background_image: torch.Tensor,
    fitted_images: list[torch.Tensor],
    output_file: str,
):
    num_images = len(original_images)
    num_garments = len(original_garments)

    # Load one image to determine the aspect ratio
    height, width = background_image.shape[1:]
    aspect_ratio = width / height
    fig_width = 15  # You can adjust this base width as needed
    fig_height = fig_width / aspect_ratio * (num_images + 1) / (num_garments + 1)

    fig, axes = plt.subplots(
        num_images + 1,
        num_garments + 1,
        figsize=(fig_width, fig_height),
    )

    # Ensure axes is a 2D array even if num_images or num_garments is 1
    if num_images == 1:
        axes = np.array([axes])
    if num_garments == 1:
        axes = np.array([axes]).T

    for i in range(num_images + 1):
        for j in range(num_garments + 1):
            ax = axes[i, j]
            if i == 0 and j == 0:
                # Top-left corner: plot background
                background_img = image_utils.tensor_to_image(background_image)
                ax.imshow(background_img, aspect="equal")
                ax.axis("off")
                ax.set_title("Background", fontsize=12)
            elif i == 0 and j > 0:
                # Top row: original images
                if j - 1 < len(original_images):
                    image = image_utils.tensor_to_image(original_images[j - 1])
                    ax.imshow(image, aspect="equal")
                    ax.axis("off")
                    if j == int(len(original_images) / 2 + 0.5):
                        ax.set_title("Images", fontsize=12)
            elif i > 0 and j == 0:
                # First column: original garments
                if i - 1 < len(original_garments):
                    garment_img = image_utils.tensor_to_image(original_garments[i - 1])
                    ax.imshow(garment_img, aspect="equal")
                    ax.axis("off")
            elif i > 0 and j > 0:
                # Fitted images
                if (i - 1) * num_garments + (j - 1) < len(fitted_images):
                    fitted = fitted_images[(i - 1) * num_garments + (j - 1)]
                    fitted_img = image_utils.tensor_to_image(fitted)
                    ax.imshow(fitted_img, aspect="equal")
                    ax.axis("off")
    # Use fig.text to add a y-label to the entire figure
    fig.text(
        0.115,
        0.5,
        "Garments",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=12,
    )
    plt.subplots_adjust(wspace=0, hspace=0.02)
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0.2, dpi=300)
    plt.close()


original_img_dir = pathlib.Path("./data/img")
original_garment_dir = pathlib.Path("./data/cloth")
fitted_img_dir = pathlib.Path("./results/matrix/img_fitted")
output_file = pathlib.Path("./results/matrix/final_comparison.jpg")
background_dir = pathlib.Path("./data/background")
background_img = "background_2032.png"

images = path_utils.read_images_from_dir(
    original_img_dir, transform=image_utils.stable_vition_image_transformation
)
garments = path_utils.read_images_from_dir(original_garment_dir)
fitted_images = path_utils.read_images_from_dir(fitted_img_dir)
backgrounds = path_utils.read_images_from_dir(background_dir)
background = backgrounds[background_img]

# Sort the dictionaries by image name and extract the sorted tensors
images = [tensor for name, tensor in sorted(images.items())]
garments = [tensor for name, tensor in sorted(garments.items())]
images_fitted = [tensor for name, tensor in sorted(fitted_images.items())]

plot_comparison_matrix(
    original_images=images,
    original_garments=garments,
    fitted_images=images_fitted,
    background_image=background,
    output_file=str(output_file),
)
