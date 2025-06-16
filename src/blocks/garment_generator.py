import datetime
import logging
import os
import re
import warnings

import torch
from PIL import Image

from building_blocks.sd3_5.sd3_infer import CONFIGS, SD3Inferencer
from src.blocks.base_block import BaseBlock

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


class GarmentGenerator(BaseBlock):
    """
    Generates garments via Stable Diffusion 3.5
    """

    def __init__(self):
        self.model_folder = "building_blocks/sd3_5/models"
        self.inferencer = SD3Inferencer()
        self.model_name = f"{self.model_folder}/sd3_medium.safetensors"  # "models/sd3-large/sd3.5_large.safetensors"
        # only required for SD3.5_large
        self.vae_file = None  # f"{self.model_folder}/sd3_vae.safetensors"
        self.controlnet = None  # f"{self.model_folder}/controlnets/sd3.5_large_controlnet_canny.safetensors"

    def unload_model(self):
        """Unload the model if it exists."""
        if not hasattr(self, "inferencer") or self.inferencer is not None:
            raise RuntimeError("Germent generator is not loaded. Can not unload.")

        del self.inferencer
        logger.info("GarmentGenerator model unloaded.")

    @torch.no_grad()
    def load_model(
        self,
        use_controlnet=False,
        device="cuda",
        verbose=False,
    ):
        config = CONFIGS.get(os.path.splitext(os.path.basename(self.model_name))[0], {})
        _shift = config.get("shift", 3)

        controlnet_ckpt = self.controlnet if use_controlnet else None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            # load weights to the inferencer
            self.inferencer.load(
                self.model_name,
                self.vae_file,
                _shift,
                controlnet_ckpt,
                self.model_folder,
                device,
                verbose,
                load_tokenizers=True,
            )
        logger.info(
            f"GarmentGenerator model loaded: {self.model_name}, "
            f"ControlNet: {controlnet_ckpt if use_controlnet else 'None'}"
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        out_dir: str,
        postfix=None,
        width=1024,
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
    ) -> list[Image.Image]:
        config = CONFIGS.get(os.path.splitext(os.path.basename(self.model_name))[0], {})
        _steps = steps or config.get("steps", 50)
        _cfg = cfg or config.get("cfg", 5)
        _sampler = sampler or config.get("sampler", "dpmpp_2m")
        skip_layer_config = CONFIGS.get(
            os.path.splitext(os.path.basename(self.model_name))[0], {}
        ).get("skip_layer_config", {})

        prompts = parse_prompts(prompt)

        sanitized_prompt = re.sub(r"[^\w\-\.]", "_", prompt)
        out_dir = os.path.join(
            out_dir,
            os.path.splitext(os.path.basename(self.model_name))[0],
            # os.path.splitext(os.path.basename(sanitized_prompt))[0][:50]
            (postfix or datetime.datetime.now().strftime("_%Y-%m-%dT%H-%M-%S")),
        )

        os.makedirs(out_dir, exist_ok=False)

        imgs = self.inferencer.gen_image(
            prompts,
            width,
            height,
            _steps,
            _cfg,
            _sampler,
            seed,
            seed_type,
            out_dir,
            controlnet_cond_image,
            init_image,
            denoise,
            skip_layer_config,
        )
        return imgs
