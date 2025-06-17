import numpy as np
from PIL import Image
import io
import torch
import json
import base64

import torch
import numpy as np
from PIL import Image
import io
import base64

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
    tensor = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # normalized float
    return tensor

def build_response_str(action: str, status: str, message: str = "", image = "") -> str:
    response = {
        "action": action,
        "status": status,
        "message": message,
        "image": image
    }
    return json.dumps(response)