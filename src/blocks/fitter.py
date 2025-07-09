import os
import sys
import time

from src.utilities.logging_utils import log_vram
import torch
from omegaconf import OmegaConf
import numpy as np
import logging

from building_blocks.StableVITON.cldm.model import create_model
from building_blocks.StableVITON.cldm.plms_hacked import PLMSSampler
from building_blocks.StableVITON.utils import tensor2img
from src.blocks.base_block import BaseBlock

sys.path.insert(0, os.path.abspath("./building_blocks/StableVITON"))
logger = logging.getLogger(__name__)


class Fitter(BaseBlock):
    """Base class for fitting models."""

    def __init__(self, ram_preload=True, run_on_gpu=True):
        super().__init__()
        self.ckp_path = "./building_blocks/StableVITON/ckpts/VITONHD.ckpt"
        self.batch_size = 1
        self.img_H = 1024
        self.img_W = 768
        self.num_denoise_steps = 50

        config_path = "./building_blocks/StableVITON/configs/VITONHD.yaml"
        self.config = OmegaConf.load(config_path)
        self.config.model.params.img_H = self.img_H
        self.config.model.params.img_W = self.img_W
        self.params = self.config.model.params

        self.model = None
        self.sampler = None

        self.shape = (4, self.img_H // 8, self.img_W // 8)

        self.is_loaded = False
        self.ram_preload = ram_preload
        self.run_on_gpu = run_on_gpu

        if ram_preload:
            self.load_model()

    def unload_model(self):
        """Unload the model if it exists."""
        if self.model is None:
            logger.info("Model not loaded. Won't unload.")
            return

        if not self.ram_preload:
            del self.model
            del self.sampler
            
        torch.cuda.empty_cache()
        self.is_loaded = False

    def load_model(self):
        """Load the model."""
        self.model = create_model(config_path=None, config=self.config)
        load_cp = torch.load(self.ckp_path, map_location="cpu")
        load_cp = load_cp["state_dict"] if "state_dict" in load_cp.keys(
        ) else load_cp
        self.model.load_state_dict(load_cp)
        self.model.eval()
        logger.info("Model loaded successfully. Load sampler now")
        self.sampler = PLMSSampler(self.model)
        self.is_loaded = True

    def __call__(self, agn_mask, cloth, cloth_mask, image, dense_pose):
        """Fit cloth"""
        if self.model is None:
            logger.warning("Model not loaded. Call load_model() first.")
            return None
        mask = agn_mask
        agn = torch.clone(image)
        agn[:, mask.squeeze() > 0] = 0.5
        raw_in = dict(
            agn=agn * 2 - 1,
            agn_mask=1 - agn_mask,
            cloth=cloth * 2 - 1,
            cloth_mask=cloth_mask,
            image=image * 2 - 1,
            image_densepose=dense_pose * 2 - 1,
        )

        if self.run_on_gpu:
            log_vram("Before Loading StableVITON model on GPU", logger)
            start = time.time()
            self.model = self.model.cuda()
            logger.info("Moved StableVITON model to GPU in %.2f seconds",
            time.time() - start)
            log_vram("After Loading StableVITON model on GPU", logger)
        
        batch = self.transform_input(raw_in)

        start = time.time()
        
        z, c = self.model.get_input(batch, self.params.first_stage_key)
        bs = z.shape[0]
        c_crossattn = c["c_crossattn"][0][:bs]
        if c_crossattn.ndim == 4:
            c_crossattn = self.model.get_learned_conditioning(c_crossattn)
            c["c_crossattn"] = [c_crossattn]
        uc_cross = self.model.get_unconditional_conditioning(bs)
        uc_full = {"c_concat": c["c_concat"], "c_crossattn": [uc_cross]}
        uc_full["first_stage_cond"] = c["first_stage_cond"]
        
        if self.run_on_gpu:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
        self.sampler.model.batch = batch

        ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
        start_code = self.model.q_sample(z, ts)

        logger.info("Took %.2f seconds from loading until start of sampling Sampling...", time.time() - start)
        start = time.time()
        
        samples, _, _ = self.sampler.sample(
            self.num_denoise_steps,
            bs,
            self.shape,
            c,
            x_T=start_code,
            verbose=False,
            eta=0.0,
            unconditional_conditioning=uc_full,
        )
        
        logger.info("Took %.2f seconds for sampling", time.time() - start)
        start = time.time()

        x_samples = self.model.decode_first_stage(samples)
        x_sample_img = tensor2img(x_samples.float())[:, :, ::-1]

        result = np.copy(x_sample_img)
        result[:, :, 0] = x_sample_img[:, :, 2]
        result[:, :, 2] = x_sample_img[:, :, 0]
        result = torch.from_numpy(result)
        
        logger.info("Took %.2f seconds after sampling", time.time() - start)

        if self.run_on_gpu:
            log_vram("Before unloading StableVITON model from GPU", logger)
            start = time.time()
            self.model = self.model.cpu()
            logger.info("Moved StableVITON model to CPU in %.2f seconds",
                time.time() - start)
            log_vram("After unloading StableVITON model from GPU", logger)
        
        return result.permute(2, 0, 1)
        # return x_sample_img[:, :, ::-1]

    def transform_input(self, raw_in):
        """Transform the input data into the format required by the model."""
        for k, v in raw_in.items():
            if type(v) is not torch.Tensor:
                logger.warning(
                    f"Warning: {k} is not a torch.Tensor, skipping transformation.")
                print(v)
            else:
                #print(f"Transforming {k} with shape {v.shape}")
                raw_in[k] = v.permute((1, 2, 0)).unsqueeze(0)
                #print("---> to shape", raw_in[k].shape)
        return raw_in
