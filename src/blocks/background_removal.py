from src.utilities import image_utils
from src.blocks.base_block import BaseBlock
from transparent_background import Remover
from torchvision import transforms
from PIL import Image, ImageOps
from diffusers import DiffusionPipeline
import gc
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(
    "./building_blocks/photo-background-generation"))


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BackgroundRemover(BaseBlock):
    """Removes background from images using photo-background-generation."""

    def __init__(self, ram_preload=False, run_on_gpu=False):
        self.model_name = "yahoo-inc/photo-background-generation"
        self.remover: Remover | None = None
        self.diffusion_pipeline: DiffusionPipeline | None = None

        self.is_loaded = False
        self.ram_preload = ram_preload
        self.run_on_gpu = run_on_gpu

        if ram_preload:
            self.load_model()

    def unload_model(self):
        """Unload the model if it exists."""
        if self.diffusion_pipeline is not None:
            del self.diffusion_pipeline
            logger.info("Background Diffusion model unloaded.")

        if self.remover is not None:
            del self.remover
            logger.info("Background Remover model unloaded.")

        gc.collect()
        torch.cuda.empty_cache()
        self.is_loaded = False

    def load_model(self, with_masking=True):
        """Load the model"""

        if with_masking:
            self.remover = Remover()
            logger.info("Remover for background masking loaded successfully.")

        self.diffusion_pipeline = DiffusionPipeline.from_pretrained(
            self.model_name, custom_pipeline=self.model_name
        )
        self.diffusion_pipeline = self.diffusion_pipeline.to(self.device)
        logger.info(
            "Diffusion pipeline for background removal loaded successfully.")
        self.is_loaded = True

    def __call__(
        self,
        img,
        prompt: str,
        subject_mask,
        results_dir=None,
        num_images=2,
        device="cuda",
        annotate_images=False,
        save_background=False,
    ) -> list[Image.Image]:
        if not hasattr(self, "diffusion_pipeline") or self.diffusion_pipeline is None:
            raise RuntimeError("Background Diffusion is not loaded.")
        result_shape = img.shape[1:]
        img = image_utils.tensor_to_image(img)
        subject_mask = image_utils.tensor_to_image(subject_mask)
        # Get foreground mask
        if subject_mask is None:
            if not hasattr(self, "remover") or self.remover is None:
                raise RuntimeError(
                    "No mask provided and also Background Remover is not loaded."
                )

            subject_mask = self.remover.process(img, type="map")
            logger.info("Foreground mask generated successfully.")

            if annotate_images:
                image_utils.add_title_to_image(
                    subject_mask,
                    title="Foreground Mask",
                    font_size=15,
                    color=(255, 255, 255),
                )
            if results_dir is not None:
                image_utils.save_image(
                    subject_mask, os.path.join(
                        results_dir, "foreground_mask.png")
                )

        # Background generation
        seed = 13
        mask = ImageOps.invert(subject_mask)
        # img = image_utils.resize_with_padding(img, (400, 400))
        generator = torch.Generator(device=device).manual_seed(seed)
        cond_scale = 1.0
        with torch.autocast(device):  # type: ignore[reportPrivateImportUsage]
            images = self.diffusion_pipeline(
                prompt=prompt,
                image=img,
                mask_image=mask,
                control_image=mask,
                num_images_per_prompt=num_images,
                generator=generator,
                num_inference_steps=20,
                guess_mode=False,
                controlnet_conditioning_scale=cond_scale,
            ).images  # type: ignore[reportCallIssue]
        logger.info("Background generated successfully.")

        # Save generated images
        for i, image in enumerate(images):
            if annotate_images:
                image = image_utils.add_title_to_image(
                    image,
                    title=f"Prompt: {prompt}, Image ({i}/{len(images) - 1})",
                    font_size=15,
                    color=(255, 255, 255),
                )
            if save_background and results_dir is not None:
                image_utils.save_image(
                    image, os.path.join(
                        results_dir, f"background_image_{i}.png")
                )

        image = image_utils.image_to_tensor(
            images[0])  # Use the first generated image
        resize_transform = transforms.Resize(result_shape)
        image = resize_transform(image)
        return image
