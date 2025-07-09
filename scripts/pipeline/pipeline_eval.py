import datetime
import logging
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
sys.path.insert(0, os.path.abspath(f"{root_dir}/building_blocks/StableVITON"))

import pathlib

import torch

from src.blocks.dense_pose import DensePose
from src.blocks.fitter import Fitter
from src.blocks.foreground_masking import ForegroundMasking
from src.blocks.harmonizer import Harmonizer
from src.blocks.image_generator import SDImageGenerator
from src.blocks.masking import Masking
from src.utilities import image_utils, path_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()

DEVICE = "cpu"


def run():
    # Input paths
    img_dir = pathlib.Path("./eval/input/imgs")
    bg_prompts_file = pathlib.Path("./eval/input/prompts/background_prompts.txt")
    garment_prompts_file = pathlib.Path(
        "./eval/input/prompts/garment_prompts_generated.csv"
    )

    max_images = 20
    start_idx = 0
    images = list(
        path_utils.read_images_from_dir(
            img_dir, transform=image_utils.stable_vition_image_transformation
        ).values()
    )[start_idx : start_idx + max_images]
    background_prompts = path_utils.read_prompts_from_file(bg_prompts_file)[
        start_idx : start_idx + max_images
    ]
    garment_prompts = path_utils.read_prompts_from_file(garment_prompts_file)[
        start_idx : start_idx + max_images
    ]

    # Output paths
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    garment_dir = pathlib.Path(f"./eval/output/{timestamp}/garments")
    background_dir = pathlib.Path(f"./eval/output/{timestamp}/backgrounds")
    fullbody_mask_dir = pathlib.Path(f"./eval/output/{timestamp}/fullbody_masks")
    agnostic_mask_dir = pathlib.Path(f"./eval/output/{timestamp}/agnostic_masks")
    garment_mask_dir = pathlib.Path(f"./eval/output/{timestamp}/garment_masks")
    denseposes_dir = pathlib.Path(f"./eval/output/{timestamp}/dense_poses")
    img_bg_replaced_dir = pathlib.Path(f"./eval/output/{timestamp}/img_bg_replaced")
    img_harmonized_dir = pathlib.Path(f"./eval/output/{timestamp}/img_harmonized")
    img_fitted_dir = pathlib.Path(f"./eval/output/{timestamp}/img_fitted")

    output_dirs = [
        garment_dir,
        background_dir,
        fullbody_mask_dir,
        agnostic_mask_dir,
        garment_mask_dir,
        denseposes_dir,
        img_bg_replaced_dir,
        img_harmonized_dir,
        img_fitted_dir,
    ]
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Stable Diffusion Image Generator...")
    image_generator = SDImageGenerator(run_on_gpu=True)
    image_generator.load_model(verbose=False)

    # generate garments
    logger.info("Generating garments...")
    garments: list[torch.Tensor] = []
    for i, garment_prompt in enumerate(garment_prompts):

        garment_prompt = (
            garment_prompt
            + " neatly hung in front of a white wall, isolated product shot, "
            "studio lighting, realistic texture, garment fully visible, "
            "photo-realistic, entire garment visible, garment centered, size m"
        )

        garment = image_generator(
            prompts=garment_prompt,
            width=768,
            height=1024,
            steps=50,
        )[0]

        garments.append(garment)
        garments_img = image_utils.tensor_to_image(garment)
        filename = str(garment_dir / f"garment_{1000 + i}.png")
        image_utils.save_image(garments_img, filename)

    # generate backgrounds
    logger.info("Generating backgrounds...")
    backgrounds: list[torch.Tensor] = []
    for i, background_prompt in enumerate(background_prompts):

        background_prompt = (
            background_prompt
            + ", photo-realistic, realistic lighting, highly detailed texture, "
            "viewed from head height, shot with a DSLR camera, "
            "soft shadows"
        )

        background = image_generator(
            prompts=background_prompt,
            width=768,
            height=1024,
            steps=50,
        )[0]

        backgrounds.append(background)
        background_img = image_utils.tensor_to_image(background)
        filename = str(background_dir / f"background_{2000 + i}.png")
        image_utils.save_image(background_img, filename)

    image_generator.unload_model()

    # Subject Masking
    logger.info("Loading LangSam Masking model...")
    masking = Masking()
    masking.load_model()

    fullbody_masks: list[torch.Tensor] = []
    agn_masks: list[torch.Tensor] = []

    for i, img in enumerate(images):
        logger.info(f"Masking image {i + 1}/{len(images)}...")
        img, fullbody, agn, mask = masking(img)
        fullbody_masks.append(fullbody)
        agn_masks.append(mask)

        # Save images
        fullbody_mask_img = image_utils.tensor_to_image(fullbody)
        agn_mask_img = image_utils.tensor_to_image(mask)

        fullbody_filename = str(fullbody_mask_dir / f"fullbody_mask_{3000 + i}.png")
        agn_filename = str(agnostic_mask_dir / f"agn_mask_{4000 + i}.png")
        image_utils.save_image(fullbody_mask_img, fullbody_filename)
        image_utils.save_image(agn_mask_img, agn_filename)

    masking.unload_model()

    #### Dense Pose ####
    logger.info("Loading DensePose model...")
    dense_pose = DensePose()
    dense_pose.load_model()

    dense_poses: list[torch.Tensor] = []
    for i, (img, fullbody) in enumerate(zip(images, fullbody_masks)):
        dense_pose_img = dense_pose(img, fullbody)  # type: ignore
        dense_poses.append(dense_pose_img)

        dense_pose_img = image_utils.tensor_to_image(dense_pose_img)
        filename = str(denseposes_dir / f"dense_pose_{5000 + i}.png")
        image_utils.save_image(dense_pose_img, filename)

    dense_pose.unload_model()

    # TODO: delete this
    garment_dir = pathlib.Path("./eval/input/cloth")
    garments = path_utils.read_images_from_dir(garment_dir)
    garments = list([tensor for name, tensor in sorted(garments.items())])[
        : start_idx + max_images
    ]

    # Garment Masking
    foreground_masking = ForegroundMasking()
    foreground_masking.load_model()

    garment_masks: list[torch.Tensor] = []
    for i, garment in enumerate(garments):
        garment_mask = foreground_masking(garment)
        garment_masks.append(garment_mask)

        garment_mask_img = image_utils.tensor_to_image(garment_mask)
        filename = str(garment_mask_dir / f"garment_mask_{6000 + i}.png")
        image_utils.save_image(garment_mask_img, filename)

    foreground_masking.unload_model()

    """
    # Garment Masking with langSam
    from lang_sam import LangSAM
    sam = LangSAM()
    garment_masks: list[torch.Tensor] = []
    for i, garment in enumerate(garments):
        image_pil = Image.fromarray((garment.permute(1, 2, 0) * 255).byte().cpu().numpy())
        text_prompt_garment = "garment with motiv."
        results = sam.predict([image_pil], [text_prompt_garment])
        garment_mask = torch.from_numpy(results[0]["masks"][0]).unsqueeze(0)

        garment_masks.append(garment_mask)

        garment_mask_img = image_utils.tensor_to_image(garment_mask)
        filename = str(garment_mask_dir / f"garment_mask_{6000 + i}.png")
        image_utils.save_image(garment_mask_img, filename)

    sam = None
    """

    # Replace Background
    logger.info("Replacing backgrounds in images...")
    img_bg_replaced: list[torch.Tensor] = []
    for i, (img, mask, background) in enumerate(
        zip(images, fullbody_masks, backgrounds)
    ):
        mask = mask.unsqueeze(0)  # Add batch dimension
        img_replaced = img * mask + background * (1 - mask)
        img_replaced = img_replaced.squeeze(0)
        img_bg_replaced.append(img_replaced)

        img_replaced_img = image_utils.tensor_to_image(img_replaced)
        filename = str(img_bg_replaced_dir / f"img_bg_replaced_{7000 + i}.png")
        image_utils.save_image(img_replaced_img, filename)

    # Harmonize Images
    logger.info("Harmonizing images...")
    harmonizer = Harmonizer()
    harmonizer.load_model()

    img_harmonized: list[torch.Tensor] = []
    img_harmonized = img_bg_replaced
    for i, (img, fullbody_mask) in enumerate(zip(img_bg_replaced, fullbody_masks)):
        img = img.unsqueeze(0)  # Add batch dimension
        fullbody_mask = fullbody_mask.unsqueeze(0)
        print(img.shape)
        harmonized = harmonizer(img, fullbody_mask)
        img_harmonized.append(harmonized)

        img_harmonized_img = image_utils.tensor_to_image(harmonized)
        filename = str(img_harmonized_dir / f"img_harmonized_{8000 + i}.png")
        image_utils.save_image(img_harmonized_img, filename)

    harmonizer.unload_model()

    # Fit Garments
    logger.info("Fitting garments to images...")
    fitter = Fitter()
    fitter.load_model()

    img_fitted: list[torch.Tensor] = []
    for i, (agn, garment, garment_mask, img, dense_pose) in enumerate(
        zip(
            agn_masks,
            garments,
            garment_masks,
            img_harmonized,
            dense_poses,
        )
    ):
        fitted = fitter(agn, garment, garment_mask, img, dense_pose)
        img_fitted.append(fitted)

        img_fitted_img = image_utils.tensor_to_image(fitted)
        filename = str(img_fitted_dir / f"img_fitted_{9000 + i}.png")
        image_utils.save_image(img_fitted_img, filename)


if __name__ == "__main__":
    run()
