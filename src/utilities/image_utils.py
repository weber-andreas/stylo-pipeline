import io
import logging
import os

import requests
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms

logger = logging.getLogger(__name__)


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))

    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (
        pad_width,
        pad_height,
        delta_width - pad_width,
        delta_height - pad_height,
    )
    return ImageOps.expand(img, padding)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a PyTorch tensor to a PIL Image.
    The tensor should be in the format (C, H, W) and normalized to [0, 1].
    """
    if tensor.dim() != 3 or tensor.shape[0] not in [1, 3]:
        raise ValueError("Input tensor must be of shape (C, H, W) with C=1 or C=3.")

    to_pil = transforms.ToPILImage()
    return to_pil(tensor)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL Image to a PyTorch tensor.
    The image should be in RGB format.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    to_tensor = transforms.ToTensor()
    return to_tensor(image)


def save_image(image: Image.Image, filename: str):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

    image.save(filename)
    logger.info(f"Image saved as {filename}")


def download_image_from_url(url: str) -> Image.Image:
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        try:
            image = Image.open(io.BytesIO(response.content))
            return image
        except Exception as e:
            raise ValueError(f"Failed to parse image content: {e}")
    else:
        raise ValueError(
            f"Failed to download image. HTTP {response.status_code}: {response.text[:200]}"
        )


def add_title_to_image(image, title, font=None, font_size=20, color=(255, 255, 255)):
    draw = ImageDraw.Draw(image)
    if font is None:
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(font, font_size)

    # Get bounding box of the text: (left, top, right, bottom)
    bbox = draw.textbbox((0, 0), title, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    width, height = image.size
    x = (width - text_width) // 2
    y = height - text_height - 10  # 10 pixels from the bottom

    # draw background rectangle
    draw.rectangle(
        [x - 5, y - 5, x + text_width + 5, y + text_height + 5],
        fill=(0, 0, 0, 128),  # semi-transparent black
    )
    draw.text((x, y), title, fill=color, font=font)

    return image


def pad_to_aspect(img: Image.Image, target_size) -> Image.Image:
    target_aspect = target_size[1] / target_size[0]  # width/height
    width, height = img.size
    current_aspect = width / height

    # Image already wide enough, no padding needed
    if current_aspect >= target_aspect:
        return img

    new_width = int(target_aspect * height)
    pad_total = new_width - width
    pad_left = pad_total // 2

    new_img = Image.new(img.mode, (new_width, height), color=(255, 255, 255))
    new_img.paste(img, (pad_left, 0))

    return new_img


# Compose pipeline with custom padding, resize, and tensor conversion
SIZE = (1024, 768)  # (height, width)
stable_vition_image_transformation = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Lambda(lambda img: pad_to_aspect(img, SIZE)),
        transforms.Resize(SIZE[1]),  # Resize height
        transforms.CenterCrop(SIZE),  # Center crop width
        transforms.ToTensor(),
    ]
)
