import logging

from src.blocks.garment_generator import GarmentGenerator

logger = logging.getLogger(__name__)


def main():
    generator = GarmentGenerator()

    generator.load_model(use_controlnet=True, device="cuda", verbose=True)
    generator(
        prompt="A futuristic garment design",
        out_dir="results/garment_generator",
    )


if __name__ == "__main__":
    main()
