import logging

import clip
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_similarity(
    image_features: torch.Tensor, text_features: torch.Tensor
) -> torch.Tensor:
    """Compute cosine similarity between image and text features."""
    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity, dot product of normalized vectors
    similarities = (image_features @ text_features.T).squeeze()
    return similarities


class ClipL:
    """OpenAS's Contrastive Language-Image Pretraining model."""

    def __init__(self, device="cuda"):
        self.model_path = (
            "building_blocks/sd3_5/models/text_encoders/clip_l.safetensors"
        )
        self.device = device
        self.model = None
        self.is_loaded = False

    def unload_model(self):
        """Unload the model if it exists."""
        if self.model is not None:
            del self.model
            logger.info("Clip model unloaded.")

        torch.cuda.empty_cache()
        self.is_loaded = False

    def load_model(self):
        """Load the model"""
        self.model, _ = clip.load("ViT-B/32", device=self.device)
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

        return clip.tokenize(texts).to(self.device)

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
