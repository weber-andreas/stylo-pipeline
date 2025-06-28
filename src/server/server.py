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
from src.server.utils import *
from src.server.pipeline_controller import PipelineController
from websockets import WebSocketServerProtocol, serve
from torchvision import transforms
from PIL import Image
import torch
from pathlib import Path
import json
import asyncio
import argparse

import logging
import os
import sys

sys.path.insert(0, os.path.abspath("./building_blocks/StableVITON"))
sys.path.insert(0, os.path.abspath("./building_blocks/sd3_5"))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")))


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
    if _active_client is not None:
        warning_msg = "Refusing new connection: another client is already connected"
        await send_action_war(ws, logger, "connect", warning_msg)

        await ws.close(
            code=1013, reason="Service Unavailable"
        )  # 1013 = Try Again Later
        return

    _active_client = ws  # register this client
    peer = ws.remote_address

    # send successfull connection status
    connect_msg = "Successfull connection!"
    await send_action_succ(ws, logger, "connect", connect_msg)

    try:
        async for message in ws:
            logger.info(
                "Current vram usage before request: %s GB",
                round(torch.cuda.memory_allocated() / 1024**3, 2),
            )

            valid, request = await check_request_data(ws, logger, message, [])
            if not valid:
                continue

            action = request["action"].lower().strip()

            logger.info("recieved action: %s", action)
            match action:
                case "upload":
                    cur_action = "upload"
                    if not await field_exist(ws, logger, request, "image", cur_action):
                        continue

                    if not await check_field_type(ws, logger, request, "image", cur_action):
                        continue

                    image_data = request["image"]
                    img_tensor = decode_tensor_from_json(image_data)

                    # Crop image to deafult aspect ratio and downsample size
                    SIZE = (1024, 768)

                    img_tensor = pad_aspect_transform(img_tensor, SIZE=SIZE)
                    _controller.set_stock_image(img_tensor, auto=True)

                    succ_msg = "Stock image uploaded successfully."
                    await send_action_succ(ws, logger, cur_action, succ_msg)

                case "background":
                    cur_action = "background"
                    # Handle background removal request
                    if not await field_exist(ws, logger, request, "prompt", cur_action):
                        continue

                    if not await check_field_type(ws, logger, request, "prompt", cur_action):
                        continue

                    prompt = request["prompt"]
                    logger.info(
                        "Background removal requested with prompt: %s", prompt)

                    bg_rm_result = _controller.remove_background(prompt)
                    if not await check_block_response(ws, logger, bg_rm_result, "background_remover", cur_action):
                        continue

                    harmonized_img = _controller.harmonize_image()

                    if not await check_block_response(ws, logger, harmonized_img, "harmonizer", cur_action):
                        continue

                    await send_action_succ(ws, logger, cur_action,
                                           "Background removed and image harmonized.")
                    continue

                case "design":
                    cur_action = "design"
                    if not await field_exist(ws, logger, request, "prompt", cur_action):
                        continue

                    if not await check_field_type(ws, logger, request, "prompt", cur_action):
                        continue

                    prompt = request["prompt"]
                    logger.info(
                        "Garment design requested with prompt: %s", prompt)
                    garment = _controller.design_garment(prompt, auto=True)

                    if not await check_block_response(ws, logger, garment, "garment_generator", cur_action):
                        continue

                    await send_action_succ(
                        ws, logger, cur_action, "Designed cloth successfully.", tensor_to_base64_png(garment))
                    continue

                case "fit":
                    cur_action = "fit"
                    fitted_img = _controller.fit_garment()

                    if not await check_block_response(ws, logger, fitted_img, "stable_viton", cur_action):
                        continue

                    await send_action_succ(
                        ws, logger, cur_action, "Garment fitting completed successfully.", tensor_to_base64_png(fitted_img))

                    continue

                case "search_garment":
                    cur_action = "search_garment"

                    if not await field_exist(ws, logger, request, "prompt", cur_action):
                        continue

                    if not await check_field_type(ws, logger, request, "prompt", cur_action):
                        continue

                    prompt = request["prompt"]

                    topk = 5
                    if "topk" in request:
                        topk = int(request["topk"])
                        logger.info("got 'topk' for search_garment: %s",
                                    request["topk"])

                    logger.info(
                        "Garment design requested with search_prompt: %s and topk: %s", prompt, str(topk))
                    # todo:: change
                    garment = _controller.search_garment(prompt, topk)

                    if not await check_block_response(ws, logger, garment, "garment_search", cur_action):
                        continue

                    await send_action_succ(ws, logger, cur_action,
                                           "search_garment successfully.", image=json.dumps(garment))
                    continue

                case "rating":
                    cur_action = "rating"

                    if not await field_exist(ws, logger, request, "rating", cur_action):
                        continue

                    fields = [
                        "usability",
                        "customizability",
                        "overall_quality",
                        "background_quality",
                        "garment_generation_quality",
                        "fitting_quality",
                    ]
                    missing_field = False
                    for f in fields:
                        if f not in request["rating"]:
                            err_msg = "Missing '" + f + "' field in rating."
                            await send_action_err(ws, logger, cur_action, err_msg)
                            missing_field = True
                            break

                    if missing_field:
                        continue

                    _controller.save_rating(request["rating"], fields, peer)
                    await send_action_succ(ws, logger, cur_action,
                                           "Successfully saved rating!")
                    continue

                case default:
                    await send_action_err(ws, logger, "unkown",
                                          "action was not found!")
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


async def check_request_data(ws, logger, message, fields):
    if isinstance(message, bytes):
        await send_action_err(ws, logger, "unkown",
                              "Binary frames not allowed for commands.")
        return (False, None)

    try:
        request = json.loads(message)

        if "action" not in request:
            error_msg = "Missing 'action' field in request."
            await send_action_err(ws, logger, "unkown", error_msg)
            return (False, None)

        return (True, request)

    except json.JSONDecodeError:
        error_msg = "Invalid JSON."
        await send_action_err(ws, logger, "unkown", error_msg)
        return (False, None)


async def field_exist(ws, logger, request, field, action):
    if field not in request:
        err_msg = "Field '{field}' in request not found!".format(field=field)
        await send_action_err(ws, logger, action, err_msg)
        return False
    return True


async def check_field_type(ws, logger, request, field, action):
    if not isinstance(request[field], str):
        err_msg = "Field '{field}' is not type String!".format(field=field)
        await send_action_err(ws, logger, action, err_msg)
        return False
    return True


async def check_block_response(ws, logger, response, block, action):
    if type(response) is str:
        err_msg = "Block: '{block}' failed: {err}".format(
            block=block, err=response)
        await send_action_err(ws, logger, action, err_msg)
        return False
    return True

# --------------------------- Entry point ----------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single‑client image WebSocket server")
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
