import logging
import os

import matplotlib.pyplot as plt
import torch
from PIL import Image

# from src.blocks.background_removal import BackgroundRemover
from src.blocks.dense_pose import DensePose
from src.blocks.fitter import Fitter
from src.blocks.masking import Masking
from src.utilities.path_utils import validate_paths_exist

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()


def main():
    logger.info("Using torch version: %s", torch.__version__)

    img_path = "./data/img/00006_00.jpg"
    cloth_path = "./data/cloth/00008_00.jpg"
    cloth_mask_path = "./data/cloth_mask/00008_00.jpg"

    paths = [img_path, cloth_path, cloth_mask_path]

    # load_cloth
    if not validate_paths_exist(paths):
        return

    #### Background Removal ####
    # gb_remover = BackgroundRemover()
    # gb_remover.load_model(device="cuda")
    # img = Image.open(img_path).convert("RGB")
    # imgs = gb_remover(
    #     img,
    #     prompt="A professional standing in a modern, well-lit office with minimalist decor and large windows",
    #     results_dir="results/background_removal",
    # )

    #### Masking and DensePose ####
    cloth = torch.from_numpy(plt.imread(cloth_path)).permute(2, 0, 1)
    cloth_mask = torch.from_numpy(plt.imread(cloth_mask_path)).unsqueeze(0)
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

    #### Fit Garment to Person ####
    fitter = Fitter()
    fitter.load_model()
    styled = fitter(
        agn=agn,
        agn_mask=mask,
        cloth=cloth,
        cloth_mask=cloth_mask,
        image=img,
        dense_pose=dense_pose_img,
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
        save_path="results/stable_viton_output.png",
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
