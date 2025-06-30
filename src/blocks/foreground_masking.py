from src.utilities import image_utils
from src.blocks.base_block import BaseBlock
from transparent_background import Remover
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(
    "./building_blocks/photo-background-generation"))


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ForegroundMasking(BaseBlock):
    """Removes the background of an image."""

    def __init__(self, ram_preload=False, run_on_gpu=False):
        self.remover = Remover()

        self.is_loaded = False
        self.ram_preload = ram_preload
        self.run_on_gpu = run_on_gpu

        if ram_preload:
            self.load_model()

    def unload_model(self):
        """Unload the model if it exists."""
        if self.remover is not None:
            del self.remover
            logger.info("Background Remover model unloaded.")

        torch.cuda.empty_cache()
        self.is_loaded = False

    def load_model(self):
        """Load the model"""
        self.remover = Remover()
        logger.info("Remover for background masking loaded successfully.")
        self.is_loaded = True

    def __call__(
        self,
        img,
    ) -> torch.Tensor:
        img = image_utils.tensor_to_image(img.cpu())
        cloth_mask = self.remover.process(img, type="map")
        logger.info("Foreground mask generated successfully.")
        return image_utils.image_to_tensor(cloth_mask)
