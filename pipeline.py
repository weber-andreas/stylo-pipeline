import logging

from src.blocks.garment_generator import GarmentGenerator
from src.blocks.masking import Masking

logger = logging.getLogger(__name__)


def main():
    """generator = GarmentGenerator()

    generator.load_model(use_controlnet=True, device="cuda", verbose=True)
    generator(
        prompt="A futuristic garment design",
        out_dir="results/garment_generator",
    )"""

    img_path = "./data/img"
    cloth_path = "./data/cloth"

    masking = Masking()
    masking.load_model()

if __name__ == "__main__":
    main()
