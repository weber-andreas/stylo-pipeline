import logging
import os
import sys

import torch

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from src.blocks.foreground_masking import ForegroundMasking
from src.blocks.image_generator import SDImageGenerator
from src.utilities import image_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()


def main():
    logger.info("Using torch version: %s", torch.__version__)

    prompts = [
        "a white elegant wall in a modern house with minimalist decor",
        "a pure white minimalist wall in a high-end apartment",
        "a clean white painted wall with soft natural lighting in a modern living room",
        "a bright white designer wall with subtle molding details in a luxury home",
        "a crisp white backdrop with smooth texture in a modern interior",
        "a pristine white wall in a sunlit Scandinavian-style home",
        "a white matte wall with gentle light reflections in a contemporary house",
        "an all-white interior wall with architectural elegance and simplicity",
        "a spotless white background with faint shadow gradient in a studio-like home setting",
        "a white elegant wall in a modern house",
    ]
    extension = ",photo-realistic, realistic lighting, highly detailed texture, viewed from head height, shot with a DSLR camera, soft shadows"

    #### Image Generation ####
    generator = SDImageGenerator(device="cpu")
    generator.load_model(verbose=True)

    for i, prompt in enumerate(prompts):
        extended_prompt = prompt + extension
        img = generator(prompts=extended_prompt)[0]
        pil_img = image_utils.tensor_to_image(img)
        image_utils.save_image(pil_img, f"results/backgrounds/background_{i:04d}.png")
    generator.unload_model()


if __name__ == "__main__":
    main()
