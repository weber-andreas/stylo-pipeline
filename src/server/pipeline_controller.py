import json
import logging
import time
from csv import DictWriter

from src.blocks.search_garment import SearchGarment
import torch

from src.blocks.background_removal import BackgroundRemover
from src.blocks.dense_pose import DensePose
from src.blocks.fitter import Fitter
from src.blocks.foreground_masking import ForegroundMasking
from src.blocks.harmonizer import Harmonizer
from src.blocks.image_generator import SDImageGenerator
from src.blocks.masking import Masking
from src.server.utils import match_tensor_size

logger = logging.getLogger(__name__)


class PipelineController:
    def __init__(self, device="cuda"):
        logger.info("Initializing PipelineController...")
        # Initialize all blocks
        logger.info("Init blocks...")
        self.masking = Masking()
        self.dense_pose = DensePose()
        self.harmonizer = Harmonizer()
        self.fitter = Fitter()
        self.cloth_masking = ForegroundMasking()
        # self.background_remover = BackgroundRemover(device=device)
        self.image_generator = SDImageGenerator()
        self.garment_searcher = SearchGarment()

        logger.info("Init cache")
        # Initialize image cache
        self.rating_file = "./results/rating_database.csv"
        self.device = device
        self.loaded_blocks = []
        self.image_cache: dict[str, torch.Tensor | None] = {
            "stock_image": None,
            "fullbody_mask": None,
            "agn_image": None,
            "agn_mask": None,
            "cloth_image": None,
            "cloth_mask": None,
            "dense_pose": None,
            "fitted_image": None,
            "harmonized_image": None,
            "background_removed_image": None,
        }
        logger.info("PipelineController initialized.")

    def load_block(self, block):
        if not block.is_loaded:
            block.load_model()
            self.loaded_blocks.append(block)
            logger.info(
                f"Block {block.__class__.__name__} loaded successfully.")
        else:
            logger.info(f"Block {block.__class__.__name__} is already loaded.")

    def unload_block(self, block):
        if block.is_loaded and not block.ram_preload:
            block.unload_model()
            logger.info(
                f"Block {block.__class__.__name__} unloaded successfully.")
            self.loaded_blocks.remove(block)
        else:
            logger.info(
                f"Block {block.__class__.__name__} is not loaded, nothing to unload."
            )

    def unload_all(self):
        logger.info("Unloading all blocks...")
        for block in self.loaded_blocks:
            logger.info(f"Unloading block: {block.__class__.__name__}")
            block.unload_model()
        self.loaded_blocks.clear()
        logger.info("All blocks unloaded.")

    def mask_gen(self):
        logger.info("Generating masks...")
        if self.image_cache["stock_image"] is None:
            return "Stock image not set. Please set the stock image first."

        self.load_block(self.masking)

        img, fullbody, agn, mask = self.masking(
            self.image_cache["stock_image"])  # type: ignore
        self.image_cache["agn_image"] = agn
        self.image_cache["agn_mask"] = mask
        self.image_cache["fullbody_mask"] = fullbody

        logger.info("Masks generated successfully.")

        self.unload_block(self.masking)
        return (img, fullbody, agn, mask)

    def search_garment(self, promt, topk):
        logger.info("Searching for garment...")
        self.load_block(self.garment_searcher)

        results = self.garment_searcher(promt, topk)
        logger.info("Garment search completed. Found: " +
                    str(len(results)) + " results.")

        self.unload_block(self.garment_searcher)
        return results

    def dense_pose_gen(self):
        logger.info("Generating dense pose...")
        if self.image_cache["stock_image"] is None:
            return "Stock image not set. Please set the stock image first."
        if self.image_cache["fullbody_mask"] is None:
            return "Fullbody mask not set. Please generate the fullbody mask first."

        self.load_block(self.dense_pose)

        dense_pose = self.dense_pose(
            self.image_cache["stock_image"], self.image_cache["fullbody_mask"]
        )
        self.image_cache["dense_pose"] = dense_pose

        logger.info("Dense pose generated successfully.")

        self.unload_block(self.dense_pose)
        return dense_pose

    def remove_background(self, prompt: str, device="cuda"):
        logger.info("Removing background...")
        if self.image_cache["stock_image"] is None:
            return "Stock image not set. Please set the stock image first."
        if self.image_cache["fullbody_mask"] is None:
            return "Fullbody mask not set. Please generate the fullbody mask first."

        image_size = self.image_cache["stock_image"][0].shape
        self.load_block(self.image_generator)
        background = self.image_generator(
            prompts=prompt,
            height=image_size[0],
            width=image_size[1],
        )[0]

        # replace background where the image is not masked
        mask = self.image_cache["fullbody_mask"].unsqueeze(0)
        image = self.image_cache["stock_image"]
        background = match_tensor_size(background, image)
        mask = match_tensor_size(mask, image)

        result = image * mask + background * (1 - mask)
        result = result.squeeze(0)

        self.image_cache["background_removed_image"] = result
        logger.info("Background removed successfully.")

        self.unload_block(self.image_generator)
        return result

    def harmonize_image(self):
        logger.info("Harmonizing image...")
        if self.image_cache["background_removed_image"] is None:
            return (
                "Background removed image not set. Please remove the background first."
            )
        if self.image_cache["fullbody_mask"] is None:
            return "Fullbody mask not set. Please generate the fullbody mask first."

        self.load_block(self.harmonizer)

        harmonized_img = self.harmonizer(
            self.image_cache["background_removed_image"].unsqueeze(0),
            self.image_cache["fullbody_mask"].unsqueeze(0),
        )
        self.image_cache["harmonized_image"] = harmonized_img

        logger.info("Image harmonized successfully.")

        self.unload_block(self.harmonizer)
        return harmonized_img

    def fit_garment(self):
        logger.info("Fitting garment to person...")
        if self.image_cache["agn_mask"] is None:
            return "AGN mask not set. Please generate the AGN mask first."
        if self.image_cache["cloth_image"] is None:
            return "Cloth image not set. Please set the cloth image first."
        if self.image_cache["cloth_mask"] is None:
            return "Cloth mask not set. Please set the cloth mask first."
        if self.image_cache["harmonized_image"] is None:
            return "Harmonized image not set. Please harmonize the image first."
        if self.image_cache["dense_pose"] is None:
            return "Dense pose not set. Please generate the dense pose first."

        self.load_block(self.fitter)

        fitted_img = self.fitter(
            agn_mask=self.image_cache["agn_mask"],
            cloth=self.image_cache["cloth_image"],
            cloth_mask=self.image_cache["cloth_mask"],
            image=self.image_cache["harmonized_image"],
            dense_pose=self.image_cache["dense_pose"],
        )
        self.image_cache["fitted_image"] = fitted_img

        logger.info("Garment fitted successfully.")

        self.unload_block(self.fitter)
        return fitted_img

    def get_cloth_mask(self):
        logger.info("Generating foreground mask...")
        if self.image_cache["cloth_image"] is None:
            return "Cloth image not set. Please set the stock image first."

        if self.image_cache["stock_image"] is None:
            return "Stock image not set. Please set the stock image first."
        self.load_block(self.cloth_masking)

        foreground_mask = self.cloth_masking(self.image_cache["cloth_image"])[
            0
        ].unsqueeze(0)

        image = self.image_cache["stock_image"][0]  # without batch dimension
        foreground_mask = match_tensor_size(foreground_mask, image)

        self.image_cache["cloth_mask"] = foreground_mask

        self.unload_block(self.cloth_masking)
        return foreground_mask

    def design_garment(self, prompt, auto=True):

        if self.image_cache["stock_image"] is None:
            return "Stock image not set. Please set the stock image first."

        self.load_block(self.image_generator)

        logger.info(
            "Current vram usage before request: %s GB",
            round(torch.cuda.memory_allocated() / 1024**3, 2),
        )

        # build promt:
        image_size = self.image_cache["stock_image"][0].shape
        prompt = (
            prompt
            + " neatly hung in front of a white wall, isolated product shot, studio lighting, realistic texture, garment fully visible, photo-realistic, entire garment visible, garmen centered, size m"
        )
        garment = self.image_generator(
            prompts=prompt,
            height=image_size[0],
            width=image_size[1],
        )[0]

        image = self.image_cache["stock_image"][0]  # without batch dimension
        garment = match_tensor_size(garment, image)

        self.image_cache["cloth_image"] = garment

        logger.info(
            "Current vram usage before request: %s GB",
            round(torch.cuda.memory_allocated() / 1024**3, 2),
        )

        self.unload_block(self.image_generator)

        if auto:
            logger.info("Auto-generating cloth masks after garment design...")
            self.get_cloth_mask()
            logger.info("Auto-generated cloth masks successfully.")

        return garment

    def set_stock_image(self, img, auto=False):
        """
        Set the stock image for the pipeline.
        """
        logger.info("Reset cache...")
        self.image_cache = {
            "stock_image": None,
            "fullbody_mask": None,
            "agn_image": None,
            "agn_mask": None,
            "cloth_image": None,
            "cloth_mask": None,
            "dense_pose": None,
            "fitted_image": None,
            "harmonized_image": None,
            "background_removed_image": None,
        }
        logger.info("Setting stock image...")
        self.image_cache["stock_image"] = img
        self.image_cache["harmonized_image"] = img
        logger.info("Stock image set successfully.")

        if auto:
            logger.info("Auto-generating masks after setting stock image...")
            res = self.mask_gen()
            if type(res) == str:
                return res
            logger.info("Auto-generated masks successfully.")
            res = self.dense_pose_gen()
            if type(res) == str:
                return res
            logger.info("Auto-generated dense pose successfully.")

    def set_stock_garment(self, garment, auto=True):
        self.image_cache["cloth_image"] = None
        self.image_cache["cloth_mask"] = None
        self.image_cache["cloth_image"] = garment

        if auto:
            logger.info(
                "Auto-generating cloth masks after setting stock garment...")
            self.get_cloth_mask()
            logger.info("Auto-generated cloth masks successfully.")

        return garment

    def save_rating(self, rating_json, fields, peer):
        if type(rating_json) == str:
            rating_json = json.loads(rating_json)

        fields = ["peer", "time"] + fields
        rating_json["peer"] = peer
        rating_json["time"] = time.time()

        with open(self.rating_file, "a") as f:
            writer = DictWriter(f, fieldnames=fields)
            writer.writerow(rating_json)
