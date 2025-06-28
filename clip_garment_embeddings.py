import pathlib

import torch
from torchvision import transforms

from src.utilities import path_utils
from src.utilities.clip_encoder import (
    ClipL,
    compute_similarity,
    get_highest_similarities,
)

RECOMPUTE_IMAGE_FEATURES = False  # Set to True to recompute image features
device = "cpu"

clip_model = ClipL(device=device)
clip_model.load_model()

image_embeddings_path = "data/clip_image_features.pt"

# Define prompt
prompt = (
    "A pink shirt with a collar and buttons, ideal for a casual "
    "or formal occasion. The shirt has a fitted silhouette and a slightly loose fit. "
)

if RECOMPUTE_IMAGE_FEATURES:
    # Folder containing images
    SIZE = (224, 224)  # (height, width)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(SIZE[1]),  # Resize height
            transforms.CenterCrop(SIZE),  # Center crop width
            transforms.ToTensor(),
        ]
    )
    image_dir = pathlib.Path("eval/input/cloth")
    images = path_utils.read_images_from_dir(image_dir, transform)
    image_tensors = []

    # get the image tensors from the dictionary
    image_tensors = list(images.values())
    image_names = list(images.keys())
    final_tensor = torch.stack(image_tensors).to(device)

    # Precompute the image embeddings
    with torch.no_grad():
        image_features = clip_model.encode_image(final_tensor)

    clip_model.save_image_embeddings(image_embeddings_path, image_features, image_names)


# Load precomputed image embeddings
image_embeddings = clip_model.load_image_embeddings(image_embeddings_path)
image_features = image_embeddings["features"]
image_names = image_embeddings["image_names"]

with torch.no_grad():
    text = clip_model.tokenize([prompt])
    text_features = clip_model.encode_text(text)
    similarities = compute_similarity(image_features, text_features)


image_similarity_map = {
    image_name: similarity.item()
    for image_name, similarity in zip(image_names, similarities)
}

# Get top match
top_idx = similarities.argmax().item()
top_score = similarities.max().item()
top_image_name = image_names[top_idx]

filtered_image_similarity_map = get_highest_similarities(
    image_similarity_map, k_top_elements=5, min_similarity=0.25
)
print(f"Prompt:\n{prompt}\n")
print(f"Similarities for each image: {filtered_image_similarity_map}\n")
print(f"Most similar image: {top_image_name} (similarity score: {top_score:.4f})")
