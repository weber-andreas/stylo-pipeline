import logging
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
sys.path.insert(0, os.path.abspath(f"{root_dir}/building_blocks/StableVITON"))


import matplotlib.pyplot as plt
import torch

from src.blocks.background_removal import BackgroundRemover
from src.blocks.dense_pose import DensePose
from src.blocks.fitter import Fitter
from src.blocks.harmonizer import Harmonizer
from src.blocks.masking import Masking
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
    subject_img = torch.from_numpy(plt.imread(img_path)).permute(2, 0, 1) / 255.0

    #### Mask LangSam ####
    masking = Masking()
    masking.load_model()
    img, fullbody, agn, mask = masking(subject_img)
    masking.unload_model()

    #### Dense Pose ####
    dense_pose = DensePose()
    dense_pose.load_model()
    dense_pose_img = dense_pose(img, fullbody)
    dense_pose.unload_model()

    #### Background Removal ####
    bg_remover = BackgroundRemover()
    bg_remover.load_model(device=DEVICE, with_masking=False)
    bg_img = bg_remover(
        img,
        prompt="golden hour by the sea, warm sunset light, calm ocean waves, distant mountain range in the background, soft clouds, peaceful atmosphere, realistic landscape",
        results_dir="results/background_removal",
        num_images=1,
        device=DEVICE,
        subject_mask=fullbody,
        annotate_images=False,
        save_background=False,
    )
    bg_remover.unload_model()

    #### Harmonize Image ####
    harmonizer = Harmonizer()
    harmonizer.load_model()
    bg_img = harmonizer(bg_img.unsqueeze(0), fullbody.unsqueeze(0))
    harmonizer.unload_model()

    #### Fit Garment to Person ####
    fitter = Fitter()
    fitter.load_model()
    styled = fitter(
        agn_mask=mask,
        cloth=cloth,
        cloth_mask=cloth_mask,
        image=bg_img,
        dense_pose=dense_pose_img,
    )
    fitter.unload_model()
    logger.info("Pipeline completed successfully.")

    vis(
        img.permute(1, 2, 0),
        styled.permute(1, 2, 0),
        title="Stable VITON Input and Output",
        save_path="results/final_output/stable_viton_output.png",
    )


def vis(input_data, output_data, title, save_path):
    """Visualize input and output data."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(input_data)
    ax[0].set_title("Input Image")
    ax[0].axis("off")

    ax[1].imshow(output_data)
    ax[1].set_title("Styled Output")
    ax[1].axis("off")

    plt.suptitle(title)

    if save_path:
        plt.savefig(save_path)


if __name__ == "__main__":
    main()
