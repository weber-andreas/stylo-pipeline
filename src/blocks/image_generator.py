import datetime
import logging
import os
import re
import warnings
import time
import torch

import src.utilities.image_utils as image_utils
from building_blocks.sd3_5.sd3_infer import CONFIGS, SD3Inferencer
from src.blocks.base_block import BaseBlock
from src.utilities.logging_utils import log_vram

logger = logging.getLogger(__name__)


def parse_prompts(prompt: str):
    """Parse prompts from a string or a file path."""
    prompts = []
    if isinstance(prompt, str):
        # If the prompt is a file path, read the file
        if os.path.splitext(prompt)[-1] == ".txt":
            with open(prompt, "r") as f:
                prompts = [l.strip() for l in f.readlines()]
        # Otherwise, treat it as a single prompt
        else:
            prompts = [prompt]
    return prompts


class SDImageGenerator(BaseBlock):
    """
    Generates images via Stable Diffusion 3.5
    """

    def __init__(self, ram_preload=True, run_on_gpu=True):
        self.model_folder = "building_blocks/sd3_5/models"
        self.inferencer = SD3Inferencer()
        self.model_name = f"{self.model_folder}/sd3_medium.safetensors"
        self.vae_file = None
        self.controlnet = None

        self.is_loaded = False
        self.ram_preload = ram_preload
        self.run_on_gpu = run_on_gpu

        if ram_preload:
            self.load_model()

    def unload_model(self):
        """Unload the model if it exists."""
        if not self.ram_preload and self.inferencer is not None:
            logger.info("Unloading GarmentGenerator model...")
            del self.inferencer

        torch.cuda.empty_cache()
        self.is_loaded = False

    @torch.no_grad()
    def load_model(
        self,
        use_controlnet=False,
        verbose=False,
    ):
        log_vram("Before loading SD3 model to CPU", logger)
        config = CONFIGS.get(os.path.splitext(
            os.path.basename(self.model_name))[0], {})
        _shift = config.get("shift", 3)

        controlnet_ckpt = self.controlnet if use_controlnet else None
        self.inferencer = SD3Inferencer()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            # load weights to the inferencer
            self.inferencer.load(
                self.model_name,
                self.vae_file,
                _shift,
                controlnet_ckpt,
                self.model_folder,
                "cpu",
                verbose,
                load_tokenizers=True,
            )
        logger.info(
            f"GarmentGenerator model loaded: {self.model_name}, "
            f"ControlNet: {controlnet_ckpt if use_controlnet else 'None'}"
        )
        log_vram("After loading SD3 model to CPU", logger)
        self.is_loaded = True

    @torch.no_grad()
    def __call__(
        self,
        prompts: str,
        width=768,
        height=1024,
        steps=40,
        cfg=CONFIGS,
        sampler="dpmpp_2m",
        seed=23,
        seed_type="random",
        controlnet_cond_image=None,
        init_image=None,
        denoise=1.0,
        skip_layer_config=None,
    ) -> list[torch.Tensor]:
        
        if self.run_on_gpu:
            log_vram("Before Loading SD3 model on GPU", logger)
            start = time.time()
            self.inferencer = self.inferencer.to("cuda")
            logger.info("Moved SD3 model to GPU in %.2f seconds",
                        time.time() - start)
            log_vram("After Loading SD3 model on GPU", logger)

        skip_layer_config = CONFIGS.get(
            os.path.splitext(os.path.basename(self.model_name))[0], {}
        ).get("skip_layer_config", {})
        parsed_prompts = parse_prompts(prompts)

        imgs = self.inferencer.gen_image(
            parsed_prompts,
            width,
            height,
            steps,
            cfg,
            sampler,
            seed,
            seed_type,
            controlnet_cond_image=controlnet_cond_image,
            init_image=init_image,
            denoise=denoise,
            skip_layer_config=skip_layer_config,
        )

        if self.run_on_gpu:
            log_vram("Before unloading SD3 model from GPU", logger)
            start = time.time()
            self.inferencer = self.inferencer.to("cpu")
            logger.info("Moved SD3 model to CPU in %.2f seconds",
                time.time() - start)
            log_vram("After unloading SD3 model from GPU", logger)

        # image to torch
        images = [image_utils.image_to_tensor(img) for img in imgs]
        print("shape of images:", [img.shape for img in images])
        return images
