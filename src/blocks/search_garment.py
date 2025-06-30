import logging
import base64

import clip
import torch
import os
from src.blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_similarity(
    image_features: torch.Tensor, text_features: torch.Tensor
) -> torch.Tensor:
    """Compute cosine similarity between image and text features."""
    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    image_features = image_features.float()  # Ensure float type
    text_features = text_features.float()  # Ensure float type
    # Compute cosine similarity, dot product of normalized vectors
    similarities = (image_features @ text_features.T).squeeze()
    return similarities


def get_highest_similarities(
    image_similarity_map: dict[str, float],
    k_top_elements: int = 5,
    min_similarity: float | None = None,
) -> list[tuple[str, float]]:
    """Rank images by similarity to the text features."""
    sim_dict = dict(
        zip(image_similarity_map["image_names"], image_similarity_map["features"]))
    sorted_items = sorted(
        sim_dict.items(), key=lambda x: x[1], reverse=True
    )
    sorted_items = sorted_items[:k_top_elements]

    if min_similarity is not None:
        sorted_items = [
            item for item in sorted_items if item[1] >= min_similarity]

    return sorted_items


class SearchGarment(BaseBlock):
    """OpenAS's Contrastive Language-Image Pretraining model."""

    def __init__(self, ram_preload=False, run_on_gpu=False):
        self.model_path = (
            "building_blocks/sd3_5/models/text_encoders/clip_l.safetensors"
        )
        self.model = None
        self.img_emb_path = 'data/clip_image_features.pt'
        self.img_path = 'data/original_viton_hd/test/cloth'

        self.is_loaded = False
        self.ram_preload = ram_preload
        self.run_on_gpu = run_on_gpu

        if ram_preload:
            self.load_model()

    def unload_model(self):
        """Unload the model if it exists."""
        if self.model is not None:
            del self.model
            logger.info("Clip model unloaded.")

        torch.cuda.empty_cache()
        self.is_loaded = False

    def load_model(self):
        """Load the model"""
        self.model, _ = clip.load("ViT-B/32")
        logger.info("Clip model loaded successfully.")
        self.is_loaded = True

    def encode_text(self, text) -> torch.Tensor:
        """Encode text using the CLIP model."""
        if not self.model:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, image_tensor) -> torch.Tensor:
        """Encode image using the CLIP model."""
        if not self.model:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def tokenize(self, texts):
        """Tokenize text using the CLIP model."""

        return clip.tokenize(texts)

    def load_image_embeddings(self, path: str) -> dict:
        """Load precomputed image embeddings from a file."""
        return torch.load(path)

    def save_image_embeddings(
        self, path: str, image_features: torch.Tensor, image_names: list
    ):
        """Save precomputed image embeddings to a file."""
        torch.save(
            {
                "features": image_features.cpu(),  # move to CPU if needed
                "image_names": image_names,
            },
            path,
        )
        logger.info(f"Image embeddings saved to {path}.")

    def __call__(self, prompt, topk, min_sim=None):
        text_features = self.encode_text(self.tokenize([prompt]).cuda())
        image_features = self.load_image_embeddings(self.img_emb_path)

        sim_scores = compute_similarity(image_features=image_features["features"],
                                        text_features=text_features.cpu())

        image_features["features"] = sim_scores
        sorted_results = get_highest_similarities(
            image_features, topk, min_sim)

        base64_images = []
        for path, score in sorted_results:
            path = os.path.join(self.img_path, path)
            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(
                    image_file.read()).decode('utf-8')
                base64_images.append(encoded_string)
        return base64_images
