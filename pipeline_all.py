import logging
import os
import sys

sys.path.insert(0, os.path.abspath("./building_blocks/StableVITON"))

import matplotlib.pyplot as plt
import torch

from src.blocks.background_removal import BackgroundRemover
from src.blocks.dense_pose import DensePose
from src.blocks.fitter import Fitter
from src.blocks.harmonizer import Harmonizer
from src.blocks.masking import Masking
from src.utilities import image_utils
from src.utilities.path_utils import validate_paths_exist

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda"


def main():
    logger.info("Using torch version: %s", torch.__version__)

    img_path = "./data/img/00006_00.jpg"
    cloth_path = "./data/cloth/00008_00.jpg"
    cloth_mask_path = "./data/cloth_mask/00008_00.jpg"

    paths = [img_path, cloth_path, cloth_mask_path]

    # load_cloth
    if not validate_paths_exist(paths):
        return

    #### Masking and DensePose ####
    cloth = torch.from_numpy(plt.imread(cloth_path)).permute(2, 0, 1) / 255.0
    cloth_mask = torch.from_numpy(plt.imread(cloth_mask_path)).unsqueeze(0) / 255.0
    print(f"Cloth shape: {cloth.shape}, Cloth mask shape: {cloth_mask.shape}")

    masking = Masking()
    masking.load_model()

    img, fullbody, agn, mask = masking(img_path)
    print("Image shape:", img.shape)
    masking.unload_model()

    dense_pose = DensePose()
    dense_pose.load_model()

    dense_pose_img = dense_pose(img, fullbody)
    print("DensePose image shape:", dense_pose_img.shape)
    dense_pose.unload_model()

    #### Background Removal ####
    print(mask.shape)
    img_mask = image_utils.tensor_to_image(fullbody)
    image_utils.save_image(img_mask, "results/background_removal/mask.png")

    bg_remover = BackgroundRemover()
    bg_remover.load_model(device=DEVICE, with_masking=False)
    # img = Image.open(img_path).convert("RGB")
    img_pil = image_utils.tensor_to_image(img)
    imgs = bg_remover(
        img_pil,
        prompt="A professional standing in a modern, well-lit office with minimalist decor and large windows, no people",
        results_dir="results/background_removal",
        num_images=2,
        device=DEVICE,
        subject_mask=img_mask,
        annotate_images=True,
    )
    img = image_utils.image_to_tensor(imgs[0])  # Use the first generated image
    bg_remover.unload_model()

    #### Harmonize Image ####
    # harmonizer = Harmonizer()
    # harmonizer.load_model()
    # img = harmonizer(img, fullbody)
    # harmonizer.unload_model()

    #### Fit Garment to Person ####
    fitter = Fitter()
    fitter.load_model()

    agn_pil = image_utils.tensor_to_image(agn)
    agn_mask_pil = image_utils.tensor_to_image(mask)
    dense_pose_pil = image_utils.tensor_to_image(dense_pose_img)
    img_pil = image_utils.tensor_to_image(img)
    image_utils.save_image(dense_pose_pil, "results/harmonizer/dense_pose_pil.png")
    image_utils.save_image(img_pil, "results/harmonizer/img.png")
    image_utils.save_image(agn_pil, "results/harmonizer/agn_pil.png")
    image_utils.save_image(agn_mask_pil, "results/harmonizer/agn_mask_pil.png")

    styled = fitter(
        agn=agn * 2 - 1,
        agn_mask=mask,
        cloth=cloth * 2 - 1,
        cloth_mask=cloth_mask,
        image=img * 2 - 1,  # Normalize to [-1, 1] for Stable VITON
        dense_pose=dense_pose_img * 2 - 1,
    )
    fitter.unload_model()
    logger.info("Pipeline completed successfully.")

    # Display the results
    stable_viton_input = {
        "img": img,
        "fullbody": fullbody,
        "agn": agn,
        "agn_mask": mask,
        "cloth": cloth,
        "cloth_mask": cloth_mask,
        "dense_pose_img": dense_pose_img,
    }
    outputs = {
        "styled": styled,
    }
    vis(
        transform_input(stable_viton_input),
        transform_input(outputs),
        title="Stable VITON Input and Output",
        save_path="results/final_output/stable_viton_output.png",
    )


def vis(input_data, output_data, title, save_path):
    """Visualize input and output data."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(input_data["img"])
    ax[0].set_title("Input Image")
    ax[0].axis("off")

    ax[1].imshow(output_data["styled"])
    ax[1].set_title("Styled Output")
    ax[1].axis("off")

    plt.suptitle(title)

    if save_path:
        plt.savefig(save_path)


def transform_input(raw_in):
    """Transform the input data into the format required by the model."""
    for k, v in raw_in.items():
        if type(v) is not torch.Tensor:
            print(f"Warning: {k} is not a torch.Tensor, skipping transformation.")
            print(v)
        else:
            print(f"Transforming {k} with shape {v.shape}")
            if len(v.shape) == 4:
                raw_in[k] = v.squeeze(dim=0)
            raw_in[k] = v.permute((1, 2, 0))
            print("---> to shape", raw_in[k].shape)
    return raw_in


if __name__ == "__main__":
    main()
