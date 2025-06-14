import logging
import os

import torch

from src.blocks.garment_generator import GarmentGenerator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()


def main():
    logger.info("Using torch version: %s", torch.__version__)

    #### Gargment Generation ####
    generator = GarmentGenerator()
    generator.load_model(use_controlnet=True, device="cpu", verbose=True)
    generator(
        prompt="A futuristic garment design",
        out_dir="results/garment_generator",
    )


if __name__ == "__main__":
    main()
