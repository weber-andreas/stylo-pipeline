import cv2
import numpy as np
import torch
from densepose import add_densepose_config
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from src.blocks.base_block import BaseBlock


class DensePose(BaseBlock):
    """Base class for fitting models."""

    def __init__(self, *args, **kwargs):
        # https://huggingface.co/bdsager/CatVTON/blob/f2d92bc453badfbcc045127c18ce6b48ceffbf28/DensePose/model_final_162be9.pkl
        super().__init__(*args, **kwargs)
        self.predictor = None
        self.cfg = get_cfg()
        add_densepose_config(self.cfg)
        self.cfg.MODEL.MASK_ON = True
        self.cfg.merge_from_file(
            "./building_blocks/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
        )
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.WEIGHTS = (
            "./building_blocks/detectron2/model_final_162be9.pkl"  # downloaded model
        )
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def unload_model(self):
        """Unload the model if it exists."""
        if self.predictor == None:
            print("DensePose not laoded. Won't unload.")
            return

        del self.predictor
        torch.cuda.empty_cache()

    def load_model(self):
        """Load the model."""
        print("Create/load DensePose Predictor...")
        self.predictor = DefaultPredictor(self.cfg)

    def __call__(self, image, full_body):
        image_np = (image.permute(1, 2, 0).numpy() * 255).astype("uint8")
        h_orig, w_orig = image_np.shape[:2]
        outputs = self.predictor(image_np)
        instances = outputs["instances"].to("cpu")

        boxes = instances.pred_boxes.tensor.cpu().numpy()
        x1, y1, x2, y2 = boxes[0]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        box_w, box_h = x2 - x1, y2 - y1

        dp = instances.pred_densepose[0]
        labels = torch.argmax(
            dp.fine_segm.squeeze(), dim=0
        )  # shape [H, W], values in [0..C-1]

        labels_np = labels.cpu().numpy()

        labels_vis = (255 * labels_np / labels_np.max()).astype(np.uint8)

        colored_labels = cv2.applyColorMap(labels_vis, cv2.COLORMAP_HSV)

        resized = cv2.resize(
            colored_labels, (box_w, box_h), interpolation=cv2.INTER_NEAREST
        )

        full = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
        full[y1:y2, x1:x2] = resized
        full_body = full_body.squeeze()

        full[full_body == False] = 0

        return (torch.from_numpy(full).float() / 255).permute(2, 0, 1)
