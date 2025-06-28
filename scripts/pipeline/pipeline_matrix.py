import copy
import logging
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import torch

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
sys.path.insert(0, os.path.abspath(f"{root_dir}/building_blocks/StableVITON"))

from src.blocks.fitter import Fitter
from src.blocks.harmonizer import Harmonizer
from src.utilities import image_utils, path_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()

DEVICE = "cpu"


def run():
    # Input paths
    img_dir = pathlib.Path("./data/img")
    fullbody_mask_dir = pathlib.Path("./data/fullbody_mask")
    cloth_dir = pathlib.Path("./data/cloth")
    cloth_mask_dir = pathlib.Path("./data/cloth_mask")
    dense_pose_dir = pathlib.Path("./data/densepose")
    agn_mask_dir = pathlib.Path("./data/agnostic_mask")
    batckgroud_dir = pathlib.Path("data/background")
    background_img = "background_2032.png"

    # Output paths
    output_dir = pathlib.Path("./results/matrix")
    background_replaced_dir = pathlib.Path(f"{output_dir}/img_bg_replaced")
    img_fitted_dir = pathlib.Path(f"{output_dir}/img_fitted")
    output_file = pathlib.Path(f"{output_dir}/final_comparison.txt")

    output_dirs = [
        output_dir,
        background_replaced_dir,
        img_fitted_dir,
    ]
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)

    images = path_utils.read_images_from_dir(
        img_dir, transform=image_utils.stable_vition_image_transformation
    )
    fullbody_masks = path_utils.read_images_from_dir(fullbody_mask_dir)
    garments = path_utils.read_images_from_dir(cloth_dir)
    garment_masks = path_utils.read_images_from_dir(cloth_mask_dir)
    dense_poses = path_utils.read_images_from_dir(dense_pose_dir)
    agn_masks = path_utils.read_images_from_dir(agn_mask_dir)
    backgrounds = path_utils.read_images_from_dir(batckgroud_dir)
    background = backgrounds[background_img]

    # Convert to list
    images = list(images.values())
    fullbody_masks = list(fullbody_masks.values())
    garments = list(garments.values())
    garment_masks = list(garment_masks.values())
    dense_poses = list(dense_poses.values())
    agn_masks = list(agn_masks.values())

    # Replace Background
    logger.info("Replacing backgrounds in images...")
    img_bg_replaced: list[torch.Tensor] = []

    for i, (img, mask) in enumerate(zip(images, fullbody_masks)):
        mask = mask.unsqueeze(0)  # Add batch dimension
        img_replaced = img * mask + background * (1 - mask)
        img_replaced = img_replaced.squeeze(0)
        img_bg_replaced.append(img_replaced)

        pil_img = image_utils.tensor_to_image(img_replaced)
        image_utils.save_image(
            pil_img, str(background_replaced_dir / f"img_bg_replaced_{i:04d}.png")
        )

    # Harmonize Images
    logger.info("Harmonizing images...")
    harmonizer = Harmonizer()
    harmonizer.load_model()

    img_harmonized: list[torch.Tensor] = []
    for i, (img, fullbody_mask) in enumerate(zip(img_bg_replaced, fullbody_masks)):
        img = img.unsqueeze(0)  # Add batch dimension
        fullbody_mask = fullbody_mask.unsqueeze(0)
        harmonized = harmonizer(img, fullbody_mask)
        img_harmonized.append(harmonized)

    harmonizer.unload_model()

    # Skip harmonization
    img_harmonized = copy.copy(img_bg_replaced)

    # Fit Garments
    logger.info("Fitting garments to images...")
    fitter = Fitter()
    fitter.load_model()

    img_fitted: list[torch.Tensor] = []
    for i, garment in enumerate(garments):
        for j, img in enumerate(img_harmonized):
            garment_mask = garment_masks[i]

            agn_mask = agn_masks[j]
            img = img_harmonized[j]
            dense_pose = dense_poses[j]

            fitted = fitter(agn_mask, garment, garment_mask, img, dense_pose)
            img_fitted.append(fitted)

            img_fitted_img = image_utils.tensor_to_image(fitted)
            filename = str(img_fitted_dir / f"img_fitted_{i:04d}_{j:04d}.png")
            image_utils.save_image(img_fitted_img, filename)

    fitter.unload_model()

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    run()
