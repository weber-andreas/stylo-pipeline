import pathlib

import pandas as pd
import torch
from torchvision import transforms

from src.utilities import image_utils, path_utils
from src.utilities.clip_encoder import ClipL

RECOMPUTE_IMAGE_FEATURES = True  # Set to True to recompute image features
device = "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

clip_model = ClipL(device=device)
clip_model.load_model()


# Specify specific prompt from the file that was generated via Llava on the ViTON test dataset
image_name = "00034_00.jpg"  # image seed for the prompt
image_prompts = pd.read_csv(
    "eval/input/prompts/garment_prompts_generated.csv",
    header=None,
    names=["image", "prompt"],
)
image_prompt_mapping = dict(zip(image_prompts["image"], image_prompts["prompt"]))
prompt = image_prompt_mapping[image_name]
text = clip_model.tokenize([prompt])
image_names = list(image_prompt_mapping.keys())

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

    for name, img in images.items():
        pil_image = image_utils.tensor_to_image(img)
        image_utils.save_image(pil_image, f"results/clip/{name}")

    image_tensors = list(images.values())[: len(image_prompt_mapping) - 1]
    final_tensor = torch.stack(image_tensors).to(device)

    # Precompute the image embeddings
    with torch.no_grad():
        image_features = clip_model.encode_image(final_tensor)

    torch.save(
        {
            "features": image_features.cpu(),  # move to CPU if needed
            "image_names": image_names,
        },
        "clip_image_features.pt",
    )

# Load precomputed image embeddings
image_features = torch.load("clip_image_features.pt")["features"]
image_names = torch.load("clip_image_features.pt")["image_names"]

with torch.no_grad():
    text_features = clip_model.encode_text(text)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity, dot product of normalized vectors
    similarities = (image_features @ text_features.T).squeeze()

image_similarity_map = {
    image_name: similarity.item()
    for image_name, similarity in zip(image_names, similarities)
}

# Get top match
top_idx = similarities.argmax().item()
top_image_name = image_names[top_idx]
top_score = similarities[top_idx].item()

print(f"Prompt:\n{prompt}\n")
print(f"Similarities for each image: {image_similarity_map}\n")
print(f"Most similar image: {top_image_name} (similarity score: {top_score:.4f})")
