import base64
import io
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def tensor_to_base64_png(tensor: torch.Tensor) -> str:
    """
    Converts a [C, H, W] tensor image to a base64-encoded PNG string.
    Assumes the input is in [0.0, 1.0] float or uint8 range.
    """
    if len(tensor.shape) != 3 or tensor.shape[0] not in (1, 3):
        raise ValueError("Expected tensor of shape [1,H,W] or [3,H,W]")

    # Convert to uint8 numpy array
    if tensor.dtype != torch.uint8:
        tensor = (tensor * 255).clamp(0, 255).byte()

    np_img = tensor.cpu().numpy()
    np_img = np.transpose(np_img, (1, 2, 0))  # [H, W, C]

    if np_img.shape[2] == 1:
        np_img = np_img.squeeze(-1)
        mode = "L"
    else:
        mode = "RGB"

    img = Image.fromarray(np_img, mode=mode)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded


def decode_tensor_from_json(img_base64_str: str) -> torch.Tensor:
    # Decode base64 image data
    image_bytes = base64.b64decode(img_base64_str)

    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_img = np.array(image)

    # Convert to tensor [C, H, W]
    tensor = (
        torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
    )  # normalized float
    return tensor


def build_response_str(action: str, status: str, message: str = "", image="") -> str:
    response = {"action": action, "status": status, "message": message, "image": image}
    return json.dumps(response)


def match_tensor_size(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Resize src tensor to match the spatial size of target."""
    if src.shape[-2:] != target.shape[-2:]:
        src = F.interpolate(
            src.unsqueeze(0),
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return src


async def send_action_succ(ws, logger, action, message, image=""):
    logger.info("Success: " + str(action) + message)
    await ws.send(build_response_str(action, "success", message, image))


async def send_action_err(ws, logger, action, message, image=""):
    logger.error("Error: " + str(action) + message)
    await ws.send(build_response_str(action, "error", message, image))


async def send_action_war(ws, logger, action, message, image=""):
    logger.warning("Warning: " + str(action) + message)
    await ws.send(build_response_str(action, "error", message, image))


def resize_to_aspect(img: Image.Image, target_size=(1024, 768)) -> Image.Image:
    img_width = img.width
    img_height = img.height

    diff_height = abs(img_height - target_size[0])
    diff_width = abs(img_width - target_size[1])

    if diff_height < diff_width:
        resize_width = int((target_size[0] / img_height) * img_width)
        img = img.resize((resize_width, target_size[0]))

    else:
        resize_height = int((target_size[1] / img_width) * img_height)
        img = img.resize((target_size[1], resize_height))

    return img


def pad_aspect_transform(img_tensor, SIZE=(1024, 768)):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Lambda(lambda img: resize_to_aspect(img, SIZE)),
            transforms.CenterCrop(SIZE),
            transforms.ToTensor(),
        ]
    )
    return transform(img_tensor)
