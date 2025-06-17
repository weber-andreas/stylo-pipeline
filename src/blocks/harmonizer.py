from building_blocks.Harmonizer.src import model
from src.blocks.base_block import BaseBlock
import torch
from building_blocks.Harmonizer.src import model
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Harmonizer(BaseBlock):
    """Base class for fitting models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ckp_path = "./building_blocks/Harmonizer/pretrained/harmonizer.pth"
        self.harmonizer = None
        self.is_loaded = False

    def unload_model(self):
        """Unload the model if it exists."""
        if self.harmonizer == None:
            logger.info("Harmonizer not loaded. Won't unload.")
            return

        del self.harmonizer
        torch.cuda.empty_cache()
        self.is_loaded = False


    def load_model(self):
        """Load the model."""
        self.harmonizer = model.Harmonizer()
        self.harmonizer = self.harmonizer.cuda()

        self.harmonizer.load_state_dict(torch.load(self.ckp_path), strict=True)
        self.harmonizer.eval()
        self.is_loaded = True

    def __call__(self, img, mask):
        """Harmonize the image with the given mask."""
        if self.harmonizer is None:
            logger.error("Harmonizer not loaded. Call load_model() first.")
            return None
        
        img = img.cuda()  # Add batch dimension
        mask = mask.cuda()
        with torch.no_grad():
            arguments = self.harmonizer.predict_arguments(img, mask)
            harmonized = self.harmonizer.restore_image(img, mask, arguments)[-1]
        return harmonized.squeeze().cpu()

        