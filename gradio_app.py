import logging
import os
import sys

import gradio as gr
import torch
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath("./building_blocks/StableVITON"))

from src.blocks.background_removal import BackgroundRemover
from src.blocks.dense_pose import DensePose
from src.blocks.fitter import Fitter
from src.blocks.harmonizer import Harmonizer
from src.blocks.masking import Masking
from src.utilities import image_utils, path_utils

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run(subject_img, garment_img, background_prompt):
    img_path = "./data/img/00006_00.jpg"
    cloth_path = "./data/cloth/00008_00.jpg"
    cloth_mask_path = "./data/cloth_mask/00008_00.jpg"
    dense_pose_path = "./data/dense_pose/00006_00.png"

    paths = [img_path, cloth_path, cloth_mask_path]

    # load_cloth
    if not path_utils.validate_paths_exist(paths):
        return

    #### Masking and DensePose ####
    cloth = torch.from_numpy(plt.imread(cloth_path)).permute(2, 0, 1) / 255.0
    cloth_mask = torch.from_numpy(plt.imread(cloth_mask_path)).unsqueeze(0) / 255.0
    dense_pose_img = (
        torch.from_numpy(plt.imread(dense_pose_path)).permute(2, 0, 1) / 255.0
    )
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
    styled_tensor = torch.from_numpy(styled).permute(2, 0, 1) / 255.0
    output_image = image_utils.tensor_to_image(styled_tensor)
    return output_image


with gr.Blocks(title="Stylo Pipeline") as app:
    with gr.Row():
        subject = gr.Image(label="Subject Image", type="filepath")
        garment = gr.Image(label="Garment Image", type="filepath")
        outputs = gr.Image(label="Final Image", type="pil")

    text_prompt = gr.Textbox(label="Background Prompt (e.g., 'forest in spring')")
    submit_btn = gr.Button("Run pipeline")

    submit_btn.click(
        fn=run,
        inputs=[subject, garment, text_prompt],
        outputs=outputs,
    )

if __name__ == "__main__":
    app.launch()
