import os
import sys

import torch
from omegaconf import OmegaConf
import numpy as np

from building_blocks.StableVITON.cldm.model import create_model
from building_blocks.StableVITON.cldm.plms_hacked import PLMSSampler
from building_blocks.StableVITON.utils import tensor2img
from src.blocks.base_block import BaseBlock

sys.path.insert(0, os.path.abspath("./building_blocks/StableVITON"))


class Fitter(BaseBlock):
    """Base class for fitting models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def unload_model(self):
        """Unload the model if it exists."""
        if self.model == None:
            print("Model not loaded. Won't unload.")
            return

        del self.model
        del self.sampler
        torch.cuda.empty_cache()
        self.is_loaded = False

    def load_model(self):
        """Load the model."""
        self.model = create_model(config_path=None, config=self.config)
        load_cp = torch.load(self.ckp_path, map_location="cpu")
        load_cp = load_cp["state_dict"] if "state_dict" in load_cp.keys() else load_cp
        self.model.load_state_dict(load_cp)
        self.model = self.model.cuda()
        self.model.eval()

        self.sampler = PLMSSampler(self.model)
        self.is_loaded = True

    def __call__(self, agn_mask, cloth, cloth_mask, image, dense_pose):
        """Fit cloth"""
        if self.model is None:
            print("Model not loaded. Call load_model() first.")
            return None

        print("agn_mask", agn_mask.shape)
        print("cloth", cloth.shape)
        print("cloth_mask", cloth_mask.shape)
        print("image", image.shape)
        print("dense_pose", dense_pose.shape)
        
        mask = agn_mask
        agn = torch.clone(image)
        agn[:, mask.squeeze() > 0] = 0.5
        raw_in = dict(
            agn=agn * 2 -1,
            agn_mask=1 - agn_mask,
            cloth=cloth * 2 - 1,
            cloth_mask=cloth_mask,
            image=image * 2 - 1,
            image_densepose=dense_pose * 2 - 1,
        )

        batch = self.transform_input(raw_in)

        z, c = self.model.get_input(batch, self.params.first_stage_key)
        bs = z.shape[0]
        c_crossattn = c["c_crossattn"][0][:bs]
        if c_crossattn.ndim == 4:
            c_crossattn = self.model.get_learned_conditioning(c_crossattn)
            c["c_crossattn"] = [c_crossattn]
        uc_cross = self.model.get_unconditional_conditioning(bs)
        uc_full = {"c_concat": c["c_concat"], "c_crossattn": [uc_cross]}
        uc_full["first_stage_cond"] = c["first_stage_cond"]
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()
        self.sampler.model.batch = batch

        ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
        start_code = self.model.q_sample(z, ts)

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

        x_samples = self.model.decode_first_stage(samples)
        x_sample_img = tensor2img(x_samples.float())[:, :, ::-1]

        result = np.copy(x_sample_img)
        result[:, :, 0] = x_sample_img[:, :, 2]
        result[:, :, 2] = x_sample_img[:, :, 0]
        return result
        #return x_sample_img[:, :, ::-1]

    def transform_input(self, raw_in):
        """Transform the input data into the format required by the model."""
        for k, v in raw_in.items():
            if type(v) is not torch.Tensor:
                print(f"Warning: {k} is not a torch.Tensor, skipping transformation.")
                print(v)
            else:
                print(f"Transforming {k} with shape {v.shape}")
                raw_in[k] = v.permute((1, 2, 0)).unsqueeze(0)
                print("---> to shape", raw_in[k].shape)
        return raw_in
