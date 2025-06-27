import csv
import pathlib
import time

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)

from src.utilities import image_utils, path_utils


def extract_answer(response):
    # Extract the answer from the response
    if "ASSISTANT:" in response:
        answer = response.split("ASSISTANT:")[-1].strip()
    else:
        answer = response.strip()
    return answer


model_id = "llava-hf/llava-1.5-7b-hf"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quant_config,
)

processor = AutoProcessor.from_pretrained(model_id)

imgs_dir = pathlib.Path("eval/input/cloth")
prompt_dir = pathlib.Path("eval/input/prompts")
imgs = path_utils.read_images_from_dir(imgs_dir)

results = dict()
for i, (img_name, img) in enumerate(imgs.items()):
    img = image_utils.tensor_to_image(img)

    prompt_text = (
        "Decribe the garment using the following template"
        "'[Color] [material] [garment type] with [sleeve/neckline], [fit/silhouette],"
        "and [special details], ideal for [style/occasion].'"
        "Do not describe the person wearing the garment, only the garment itself."
    )
    prompt = "USER: <image>\n" f"{prompt_text}\n" "ASSISTANT:"

    # Stop time
    start_time = time.time()
    inputs = processor(img, prompt, return_tensors="pt").to(model.device, torch.float16)
    output = model.generate(**inputs, max_new_tokens=200)
    response = processor.decode(output[0], skip_special_tokens=True)
    response = extract_answer(response)
    end_time = time.time()  # End time

    print(
        f"Processing time for image {img_name}: {(end_time - start_time):.2f} seconds"
    )
    results[img_name] = response

    # write the response to a text file
    # create file if not exists and generate


output_file = prompt_dir / "garment_prompts_generated.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    for img_name, response in results.items():
        writer.writerow([img_name, response])
print(f"Results saved to {output_file}")
