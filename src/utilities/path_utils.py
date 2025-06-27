"""Utility functions for path operations."""

import logging
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath("./building_blocks/StableVITON"))
import pathlib

import matplotlib.pyplot as plt
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)


def validate_paths_exist(paths: List[str]) -> bool:
    """Check if all paths exist."""
    for path in paths:
        if not os.path.exists(path):
            logger.error("Path does not exist: %s", path)
            return False
    return True


def read_images_from_dir(
    img_dir: pathlib.Path, transform: Optional[transforms.Compose] = None
) -> Dict[str, torch.Tensor]:
    """Read all images from a directory and return them as a list of tensors."""
    logger.info("Reading images from directory: %s", img_dir)
    loaded_imgs = {}
    for img_name in sorted(os.listdir(img_dir)):
        if img_name.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(img_dir, img_name)
            img = plt.imread(img_path)
            img_tensor = torch.from_numpy(img.copy())
            img_tensor = (
                img_tensor.permute(2, 0, 1) / 255.0
            )  # Convert to CxHxW and normalize
            if transform:
                img_tensor = transform(img_tensor)
            loaded_imgs[img_name] = img_tensor
    return loaded_imgs


def parse_description_file(file_path):
    """
    Parses a file with one description per line.
    """
    descriptions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            desc = line.strip()
            if desc:
                descriptions.append(desc)
    return descriptions


def parse_csv_file(file_path):
    """
    Parses a CSV-like file and extracts only the description part (ignores filename).
    Assumes format: filename,"description"
    """
    descriptions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                desc = parts[1].strip().strip('"')
                if desc:
                    descriptions.append(desc)
    return descriptions


def read_prompts_from_file(file_path: pathlib.Path) -> List[str]:
    """Read prompts from a text file and return them as a list of strings."""
    if file_path.suffix == ".csv":
        return parse_csv_file(file_path)
    elif file_path.suffix in [".txt", ".md"]:
        return parse_description_file(file_path)
    else:
        logger.error("Unsupported file format: %s", file_path.suffix)
        return []
