import logging
import os

import torch

#from src.blocks.garment_generator import GarmentGenerator
from src.blocks.masking import Masking
from src.blocks.dense_pose import DensePose

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()


def main():
    logger.info("Using torch version: %s", torch.__version__)
    
    """
    #### Gargment Generation ####
    generator = GarmentGenerator()
    generator.load_model(use_controlnet=True, device="cpu", verbose=True)
    generator(
        prompt="A futuristic garment design",
        out_dir="results/garment_generator",
    )"""

    img_path = "./data/img/00006_00.jpg"
    cloth_path = "./data/cloth/00008_00.jpg"
    dense_pose_path = "./data/densepose/00006_00.jpg"

    masking = Masking()
    masking.load_model()

    img, fullbody, agn, mask = masking(img_path)

    masking.unload_model()

    dense_pose = DensePose()
    dense_pose.load_model()

    dense_pose_img = dense_pose(img, fullbody)

    dense_pose.unload_model()
    

if __name__ == "__main__":
    main()
