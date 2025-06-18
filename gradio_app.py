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
from src.utilities import image_utils

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"


def run(subject_path, cloth_path, cloth_mask_path, background_prompt):

    paths = [subject_path, cloth_path, cloth_mask_path]

    # load_cloth
    # if not path_utils.validate_paths_exist(paths):
    #     return

    #### Masking and DensePose ####
    cloth = torch.from_numpy(plt.imread(cloth_path)).permute(2, 0, 1) / 255.0
    cloth_mask = torch.from_numpy(plt.imread(cloth_mask_path)).unsqueeze(0) / 255.0
    subject = torch.from_numpy(plt.imread(subject_path)).permute(2, 0, 1) / 255.0

    #### Mask LangSam ####
    masking = Masking()
    masking.load_model()
    img, fullbody, agn, mask = masking(subject)
    masking.unload_model()

    #### Dense Pose ####
    dense_pose = DensePose()
    dense_pose.load_model()
    dense_pose_img = dense_pose(img, fullbody)
    dense_pose.unload_model()

    #### Background Removal ####
    bg_remover = BackgroundRemover()
    bg_remover.load_model(device=DEVICE, with_masking=False)
    img_mask = image_utils.tensor_to_image(fullbody)
    img_pil = image_utils.tensor_to_image(img)
    imgs = bg_remover(
        img_pil,
        prompt=background_prompt,  # "star galaxy, realistic, star wars!!!!! we want yedis, planet",
        results_dir="results/background_removal",
        num_images=1,
        device=DEVICE,
        subject_mask=img_mask,
        annotate_images=False,
        save_background=False,
    )
    bg_img = image_utils.image_to_tensor(imgs[0])  # Use the first generated image
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

    styled_tensor = torch.from_numpy(styled).permute(2, 0, 1) / 255.0
    output_image = image_utils.tensor_to_image(styled_tensor)
    return output_image


with gr.Blocks(title="Stylo Pipeline") as app:
    with gr.Row():
        subject = gr.Image(label="Subject Image", type="filepath")
        cloth = gr.Image(label="Garment Cloth", type="filepath")
        cloth_mask = gr.Image(label="Garment Mask", type="filepath")
        output = gr.Image(label="Final Image", type="pil")

    text_prompt = gr.Textbox(label="Background Prompt (e.g., 'forest in spring')")
    submit_btn = gr.Button("Run pipeline")

    submit_btn.click(
        fn=run,
        inputs=[subject, cloth, cloth_mask, text_prompt],
        outputs=output,
    )

    examples = [
        [
            os.path.join(os.path.dirname(__file__), "data", "img", "00006_00.jpg"),
            os.path.join(os.path.dirname(__file__), "data", "cloth", "00008_00.jpg"),
            os.path.join(
                os.path.dirname(__file__), "data", "cloth_mask", "00008_00.jpg"
            ),
        ],
    ]

    gr.Examples(
        examples=examples,
        inputs=[subject, cloth, cloth_mask, text_prompt],
        outputs=output,
    )

if __name__ == "__main__":
    app.launch()
