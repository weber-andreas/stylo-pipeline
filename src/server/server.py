"""
Single‑client Pipeline WebSocket Server
-----------------------------------
This server handles a single client connection at a time, allowing for image processing tasks such as uploading images, removing backgrounds, and fitting garments. It uses the `websockets` library for WebSocket communication and the `PipelineController` class to manage the image processing pipeline.
-----------------------------------
Possible actions:
- `LIST`: List available actions.
- `UPLOAD`: Upload an image for processing.
- `BACKGROUND`: Remove the background from the uploaded image.
- `DESIGN`: Placeholder for design functionality (not yet implemented).
- `FIT`: Fit a garment to the uploaded image.
-----------------------------------
Requirements:
- `websockets`
- `Pillow`
- `torch`
- `numpy`
-----------------------------------
Usage:
python server.py --host localhost --port 8765
-----------------------------------
This will start the server on the specified host and port, allowing clients to connect and interact with the image processing pipeline.

"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.abspath("./building_blocks/StableVITON"))
sys.path.insert(0, os.path.abspath("./building_blocks/sd3_5"))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import argparse
import asyncio
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from websockets import WebSocketServerProtocol, serve

from src.server.pipeline_controller import PipelineController
from src.server.utils import (
    build_response_str,
    decode_tensor_from_json,
    tensor_to_base64_png,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

# --------------------------- Global state ---------------------------------- #
_active_client: WebSocketServerProtocol | None = None  # currently connected client
device = "cpu"
_controller: PipelineController = PipelineController(device=device)

# ------------------------- Helper functions -------------------------------- #


async def _handle_client(ws: WebSocketServerProtocol):
    global _active_client
    global _controller

    # Reject if another client is already connected
    # if _active_client and _active_client.open:
    if _active_client is not None:
        logger.warning(
            "Refusing new connection: another client is active from %s",
            _active_client.remote_address,
        )

        error_connect_response = build_response_str(
            "connect", "error", "Another client is already connected. Try again later."
        )
        await ws.send(error_connect_response)
        await ws.close(
            code=1013, reason="Service Unavailable"
        )  # 1013 = Try Again Later
        return

    _active_client = ws  # register this client
    peer = ws.remote_address
    logger.info("Client %s connected", peer)

    succ_connect_response = build_response_str(
        "connect", "success", "Successfull connection!"
    )
    await ws.send(succ_connect_response)

    try:
        async for message in ws:
            if isinstance(message, bytes):
                # Only text commands are accepted; ignore binary frames
                await ws.send("ERROR: Binary frames not allowed for commands.")
                continue

            try:
                request = json.loads(message)
                if "action" not in request:
                    await ws.send("ERROR: Missing 'action' field in request.")
                    continue

            except json.JSONDecodeError:
                await ws.send("ERROR: Invalid JSON message.")
                continue

            action = request["action"].lower().strip()

            logger.info(
                "Current vram usage before request: %s GB",
                round(torch.cuda.memory_allocated() / 1024**3, 2),
            )

            logger.info("recieved action: %s", action)
            match action:
                case "list":
                    await ws.send(
                        build_response_str(
                            "list",
                            "success",
                            "Available: LIST, UPLOAD, BACKGROUND, DESIGN, FIT",
                        )
                    )
                    continue

                case "upload":
                    if "image" not in request:
                        logger.error("Missing 'image' field in upload request.")
                        await ws.send(
                            build_response_str(
                                "upload",
                                "error",
                                "Missing 'image' field in upload request.",
                            )
                        )
                        continue
                    if not isinstance(request["image"], str):
                        logger.error("Invalid 'image' field type in upload request.")
                        await ws.send(
                            build_response_str(
                                "upload",
                                "error",
                                "'image' field must be a base64-encoded string.",
                            )
                        )
                        continue

                    image_data = request["image"]
                    img_tensor = decode_tensor_from_json(image_data)

                    # resize dividable by four
                    # width, height = img_tensor.shape[1], img_tensor.shape[2]
                    # width += width % 8
                    # height += height % 8
                    # resize = transforms.Resize((width, height))
                    # img_tensor = resize(img_tensor)

                    # Crop image to deafult aspect ratio and downsample size

                    SIZE = (1024, 768)  # (height, width)

                    def pad_to_aspect(
                        img: Image.Image, target_size=SIZE
                    ) -> Image.Image:
                        target_aspect = target_size[1] / target_size[0]  # width/height
                        width, height = img.size
                        current_aspect = width / height

                        # Image already wide enough, no padding needed
                        if current_aspect >= target_aspect:
                            return img

                        new_width = int(target_aspect * height)
                        pad_total = new_width - width
                        pad_left = pad_total // 2

                        new_img = Image.new(
                            img.mode, (new_width, height), color=(255, 255, 255)
                        )
                        new_img.paste(img, (pad_left, 0))

                        return new_img

                    # Compose pipeline with custom padding, resize, and tensor conversion
                    transform = transforms.Compose(
                        [
                            transforms.ToPILImage(),
                            transforms.Lambda(lambda img: pad_to_aspect(img, SIZE)),
                            transforms.Resize(SIZE[1]),  # Resize height
                            transforms.CenterCrop(SIZE),  # Center crop width
                            transforms.ToTensor(),
                        ]
                    )

                    img_tensor = transform(img_tensor)
                    print(f"Image tensor shape after upload: {img_tensor.shape}")

                    _controller.set_stock_image(img_tensor, auto=True)

                    logger.info("Stock image uploaded and set by %s", peer)
                    await ws.send(
                        build_response_str(
                            "upload", "success", "Stock image uploaded successfully."
                        )
                    )

                case "background":
                    # Handle background removal request
                    if "prompt" not in request:
                        logger.error("Missing 'prompt' field for background removal.")
                        await ws.send(
                            build_response_str(
                                "background",
                                "error",
                                "Missing 'prompt' field for background removal.",
                            )
                        )
                        continue

                    prompt = request["prompt"]

                    logger.info("Background removal requested with prompt: %s", prompt)
                    bg_rm_result = _controller.remove_background(prompt)
                    if type(bg_rm_result) is str:
                        logger.error("Background removal failed: %s", bg_rm_result)
                        await ws.send(
                            build_response_str("background", "error", bg_rm_result)
                        )
                        continue

                    harmonized_img = _controller.harmonize_image()
                    if type(harmonized_img) is str:
                        logger.error("Image harmonization failed: %s", harmonized_img)
                        await ws.send(
                            build_response_str("harmonize", "error", harmonized_img)
                        )
                        continue

                    await ws.send(
                        build_response_str(
                            "background",
                            "success",
                            "Background removed and image harmonized.",
                            tensor_to_base64_png(harmonized_img),
                        )
                    )
                    logger.info(
                        "Background removal and harmonization completed successfully and sent to %s",
                        peer,
                    )
                    continue

                case "design":
                    if "prompt" not in request:
                        logger.error("Missing 'prompt' field for designer.")
                        await ws.send(
                            build_response_str(
                                "design",
                                "error",
                                "Missing 'prompt' field for designer.",
                            )
                        )
                        continue

                    prompt = request["prompt"]
                    logger.info("Garment design requested with prompt: %s", prompt)
                    garment = _controller.design_garment(prompt, auto=True)

                    print(garment)
                    print(garment.shape)
                    if type(garment) is str:
                        logger.error("Garment design failed: %s", garment)
                        await ws.send(build_response_str("design", "error", garment))
                        continue

                    await ws.send(
                        build_response_str(
                            "design",
                            "success",
                            "Designed cloth successfully.",
                            tensor_to_base64_png(garment),
                        )
                    )
                    logger.info("Designed garment successfully and sent to %s", peer)
                    continue

                case "fit":
                    fitted_img = _controller.fit_garment()
                    if type(fitted_img) is str:
                        logger.error("Garment fitting failed: %s", fitted_img)
                        await ws.send(
                            build_response_str("fit", "error", message=fitted_img)
                        )
                        continue

                    await ws.send(
                        build_response_str(
                            "fit",
                            "success",
                            "Garment fitting completed successfully.",
                            tensor_to_base64_png(fitted_img),
                        )
                    )
                    logger.info("Garment fitting completed and sent to %s", peer)
                    continue

                case "rating":
                    if "rating" not in request:
                        logger.error("Missing 'rating' field in rating.")
                        await ws.send(
                            build_response_str(
                                "rating",
                                "error",
                                message="Missing 'rating' field in rating.",
                            )
                        )
                        continue
                    fields = [
                        "usability",
                        "customizability",
                        "overall_quality",
                        "background_quality",
                        "garment_generation_quality",
                        "fitting_quality",
                    ]
                    for f in fields:
                        if f not in request["rating"]:
                            logger.error("Missing '" + f + "' field in rating.")
                            await ws.send(
                                build_response_str(
                                    "rating",
                                    "error",
                                    message="Missing '" + f + "' field in rating.",
                                )
                            )
                            continue
                    _controller.save_rating(request["rating"], fields, peer)

                case default:
                    logger.error("Unkown action: %s", action)
                    await ws.send(
                        build_response_str(
                            "error",
                            "error",
                            f"Unknown action: {action}. Use LIST, UPLOAD, BACKGROUND, DESIGN, or FIT.",
                        )
                    )
                    continue

    except Exception as exc:  # noqa: BLE001
        logger.exception("Error while serving client %s: %s", peer, exc)
        logger.error("Unloading all blocks due to error with client %s", peer)
        _controller.unload_all()  # Ensure all blocks are unloaded on error
    finally:
        # Clean up
        await ws.close()
        if _active_client is ws:
            _active_client = None
        logger.info("Client %s disconnected", peer)


# --------------------------- Entry point ----------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Single‑client image WebSocket server")
    parser.add_argument(
        "--host", default="localhost", help="Host to bind (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="Port to bind (default: 8765)"
    )
    args = parser.parse_args()

    async def startup() -> None:
        async with serve(
            lambda ws: _handle_client(ws),
            args.host,
            args.port,
            max_size=10 * 1024 * 1024,
            ping_interval=None,
        ):
            logger.info("Server ready on http://%s:%d", args.host, args.port)
            await asyncio.Future()  # run forever

    try:
        asyncio.run(startup())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user – shutting down.")


if __name__ == "__main__":
    with torch.no_grad():
        # Ensure that the GPU memory is cleared before starting the server
        torch.cuda.empty_cache()
        main()
