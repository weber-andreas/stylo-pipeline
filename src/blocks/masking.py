import logging

import numpy as np
import torch
from lang_sam import LangSAM
from PIL import Image

from src.blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Masking(BaseBlock):
    """Base class for fitting models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint = "./building_blocks/sam/checkpoints/sam2.1_hiera_tiny.pt"
        self.model_cfg = "./building_blocks/sam/configs/sam2.1/sam2.1_hiera_t.yaml"
        self.lang_sam: LangSAM | None = None
        self.is_loaded = False

    def unload_model(self):
        """Unload the model if it exists."""
        if self.lang_sam is None:
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

    def __call__(
        self, img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """
        Mask the given image. Input image should be a path.
        Return (img, full_body, agn_mask, mask)
        """
        if self.lang_sam is None:
            logger.error("LangSam not loaded. Call load_model() first.")
            return None

        # image_pil = Image.open(img).convert("RGB")
        # image_tensor = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1) / 255
        image_pil = Image.fromarray((img.permute(1, 2, 0) * 255).byte().cpu().numpy())
        image_tensor = img

        valid_result = lambda x: x and "masks" in x[0] and len(x[0]["masks"]) > 0

        text_prompt_person = "person."
        results = self.lang_sam.predict([image_pil], [text_prompt_person])
        if not valid_result(results):
            person_mask = np.zeros(image_tensor.shape[1:], dtype=np.uint8)
            logger.warning("No person mask found. Returning empty mask.")
        else:
            person_mask = results[0]["masks"][0]

        text_prompt_pants = "pants."
        results = self.lang_sam.predict([image_pil], [text_prompt_pants])
        if not valid_result(results):
            pants_mask = np.zeros(image_tensor.shape[1:], dtype=np.uint8)
            logger.warning("No pants mask found. Returning empty mask.")
        else:
            pants_mask = results[0]["masks"][0]

        text_prompt_shirt = "shirt."
        results = self.lang_sam.predict([image_pil], [text_prompt_shirt])
        if not valid_result(results):
            shirt_mask = np.zeros(image_tensor.shape[1:], dtype=np.uint8)
            logger.warning("No shirt mask found. Returning empty mask.")
        else:
            shirt_mask = results[0]["masks"][0]

        # should be two hands
        text_prompt_hand = "hand."
        results = self.lang_sam.predict([image_pil], [text_prompt_hand])
        if not valid_result(results):
            hand_mask = np.zeros(image_tensor.shape[1:], dtype=np.uint8)
            logger.warning("No hand mask found. Returning empty mask.")
        else:
            hand_mask = results[0]["masks"][0]
            if len(results[0]["masks"]) > 1:
                hand_mask += results[0]["masks"][1]

        text_prompt_face = "face"
        results = self.lang_sam.predict([image_pil], [text_prompt_face])
        if not valid_result(results):
            face_mask = np.zeros(image_tensor.shape[1:], dtype=np.uint8)
            logger.warning("No face mask found. Returning empty mask.")
        else:
            face_mask = results[0]["masks"][0]

        text_prompt_hair = "hair"
        results = self.lang_sam.predict([image_pil], [text_prompt_hair])
        if not valid_result(results):
            hair_mask = np.zeros(image_tensor.shape[1:], dtype=np.uint8)
            logger.warning("No hair mask found. Returning empty mask.")
        else:
            hair_mask = results[0]["masks"][0]

        text_prompt_neck = "neck"
        results = self.lang_sam.predict([image_pil], [text_prompt_neck])
        if not valid_result(results):
            neck_mask = np.zeros(image_tensor.shape[1:], dtype=np.uint8)
            logger.warning("No neck mask found. Returning empty mask.")
        else:
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
