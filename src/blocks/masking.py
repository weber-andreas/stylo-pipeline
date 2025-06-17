import numpy as np
import torch
from lang_sam import LangSAM
from PIL import Image

from src.blocks.base_block import BaseBlock
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Masking(BaseBlock):
    """Base class for fitting models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint = "./building_blocks/sam/checkpoints/sam2.1_hiera_tiny.pt"
        self.model_cfg = "./building_blocks/sam/configs/sam2.1/sam2.1_hiera_t.yaml"
        self.lang_sam = None
        self.is_loaded = False

    def unload_model(self):
        """Unload the model if it exists."""
        if self.lang_sam == None:
            logger.info("Lang Sam not loaded. Won't unload.")
            return

        # del self.predictor
        del self.lang_sam
        torch.cuda.empty_cache()
        self.is_loaded = False

    def load_model(self):
        """Load the model."""
        self.lang_sam = LangSAM()
        self.is_loaded = True

    def __call__(self, img):
        """
        Mask the given image. Input image should be a path.
        Return (img, full_body, agn_mask, mask)
        """
        if self.lang_sam is None:
            logger.error("LangSam not loaded. Call load_model() first.")
            return None

        #image_pil = Image.open(img).convert("RGB")
        #image_tensor = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1) / 255
        image_pil = Image.fromarray((img.permute(1, 2, 0) * 255).byte().cpu().numpy())
        image_tensor = img

        text_prompt_person = "person."
        results = self.lang_sam.predict([image_pil], [text_prompt_person])
        person_mask = results[0]["masks"][0]

        text_prompt_pants = "pants."
        results = self.lang_sam.predict([image_pil], [text_prompt_pants])
        pants_mask = results[0]["masks"][0]

        text_prompt_shirt = "shirt."
        results = self.lang_sam.predict([image_pil], [text_prompt_shirt])
        shirt_mask = results[0]["masks"][0]

        # should be two hands
        text_prompt_hand = "hand."
        results = self.lang_sam.predict([image_pil], [text_prompt_hand])
        hand_mask = results[0]["masks"][0]
        hand_mask += results[0]["masks"][1]

        text_prompt_face = "face"
        results = self.lang_sam.predict([image_pil], [text_prompt_face])
        face_mask = results[0]["masks"][0]

        text_prompt_hair = "hair"
        results = self.lang_sam.predict([image_pil], [text_prompt_hair])
        hair_mask = results[0]["masks"][0]

        text_prompt_neck = "neck."
        results = self.lang_sam.predict([image_pil], [text_prompt_neck])
        neck_mask = results[0]["masks"][0]

        person_mask = torch.from_numpy(person_mask).unsqueeze(0)
        pants_mask = torch.from_numpy(pants_mask).unsqueeze(0)
        shirt_mask = torch.from_numpy(shirt_mask).unsqueeze(0)
        hand_mask = torch.from_numpy(hand_mask).unsqueeze(0)
        face_mask = torch.from_numpy(face_mask).unsqueeze(0)
        hair_mask = torch.from_numpy(hair_mask).unsqueeze(0)
        neck_mask = torch.from_numpy(neck_mask).unsqueeze(0)

        mask = torch.clone(person_mask)
        mask[pants_mask > 0] = 0
        mask[hand_mask > 0] = 0
        mask[face_mask > 0] = 0
        mask[hair_mask > 0] = 0

        agn_mask = torch.clone(image_tensor)
        agn_mask[:, mask[0] > 0] = 0.5

        return image_tensor, person_mask, agn_mask, mask
