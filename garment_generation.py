import logging
import os

import torch

from src.blocks.foreground_masking import ForegroundMasking
from src.blocks.image_generator import SDImageGenerator
from src.utilities import image_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()


def main():
    logger.info("Using torch version: %s", torch.__version__)

    #### Gargment Generation ####
    generator = SDImageGenerator()
    generator.load_model(use_controlnet=True, device="cpu", verbose=True)
    garments = generator(
        prompts="A white polo shirt with red stripes on the collar and sleeve cuff neatly hung in front of a white wall, isolated product shot, studio lighting, realistic texture, garment fully visible, photo-realistic, entire garment visible, garmen centered, size m",
    )

    single_garment = garments[0]
    #### Remove Background of Grament ####
    foreground_masking = ForegroundMasking()
    foreground_masking.load_model()
    garment_mask = foreground_masking(single_garment)
    foreground_masking.unload_model()
    garment_mask_img = image_utils.tensor_to_image(garment_mask)
    image_utils.save_image(garment_mask_img, "results/garment_masking/garment_mask.png")


if __name__ == "__main__":
    main()
