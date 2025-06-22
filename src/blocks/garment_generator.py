import datetime
import logging
import os
import re
import warnings

import torch
from PIL import Image

import src.utilities.image_utils as image_utils
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


class SDImageGenerator(BaseBlock):
    """
    Generates images via Stable Diffusion 3.5
    """

    def __init__(self, device):
        self.device = device
        self.model_folder = "building_blocks/sd3_5/models"
        self.inferencer = SD3Inferencer()
        self.model_name = f"{self.model_folder}/sd3_medium.safetensors"  # "models/sd3-large/sd3.5_large.safetensors"
        # only required for SD3.5_large
        self.vae_file = None  # f"{self.model_folder}/sd3_vae.safetensors"
        self.controlnet = None  # f"{self.model_folder}/controlnets/sd3.5_large_controlnet_canny.safetensors"
        self.is_loaded = False

    def unload_model(self):
        """Unload the model if it exists."""
        if self.inferencer is not None:
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
        config = CONFIGS.get(os.path.splitext(os.path.basename(self.model_name))[0], {})
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
                self.device,
                verbose,
                load_tokenizers=True,
            )
        logger.info(
            f"GarmentGenerator model loaded: {self.model_name}, "
            f"ControlNet: {controlnet_ckpt if use_controlnet else 'None'}"
        )
        self.is_loaded = True

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        out_dir: str,
        postfix=None,
        width=1024,
        height=768,
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

        # image to torch
        images = [image_utils.image_to_tensor(img) for img in imgs]
        print("shape of images:", [img.shape for img in images])
        return images
