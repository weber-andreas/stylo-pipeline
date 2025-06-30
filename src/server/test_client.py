"""
Basic WebSocket Client for Pipeline Server
------------------------------------------
Sends JSON-based commands to the server and handles base64-encoded
image responses. This matches the protocol defined in the provided
server (`server.py`).

Usage:
    python pipeline_client.py --host localhost --port 8765

Dependencies:
    pip install websockets pillow
"""
import asyncio
import websockets
import base64
import json
import argparse
from PIL import Image
import io
import cv2
import time
from random import randrange


async def send_action(ws, action, **kwargs):
    payload = {"action": action, **kwargs}
    await ws.send(json.dumps(payload))
    response = await ws.recv()
    return json.loads(response)


def display_base64_image(b64_str):
    img_data = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_data))
    img.save("src/server/example_transfers/example_transfers_" +
             str(time.time()) + ".png")


async def main(uri):
    async with websockets.connect(uri, max_size=10 * 1024 * 1024, ping_interval=None) as ws:
        """async for message in ws:
            print("Received message:", message)
            #close connection if server sends a close message
            if isinstance(message, str) and message.startswith("ERROR: Another client is already connected. Try again later."):
                print("Server requested close:", message)
                await ws.close()
                return"""

        rating = {
            'usability': randrange(1, 11),
            'customizability': randrange(1, 11),
            'overall_quality': randrange(1, 11),
            'background_quality': randrange(1, 11),
            'garment_generation_quality': randrange(1, 11),
            'fitting_quality': randrange(1, 11)
        }
        """        await send_action(ws, "rating", rating=json.dumps(rating))
        print("was sent")"""

        print("Connected to server.")
        print(await ws.recv())

        with open("src/server/example_transfers/happy-young-man-standing-over-white-background-KCKEH1.jpg", "rb") as image_file:
            data = base64.b64encode(image_file.read()).decode("utf-8")
            resp = await send_action(ws, "UPLOAD", image=data)
            print("Upload response:", resp)

        resp = await send_action(ws, "design", prompt="yellow polo shirt")
        print(resp["image"])

        """# 2. Upload an image
        with open("src/server/example_transfers/happy-young-man-standing-over-white-background-KCKEH1.jpg", "rb") as image_file:
            data = base64.b64encode(image_file.read())
        print(data)

        resp = await send_action(ws, "UPLOAD", image=data)
        print("Upload response:", resp)

        # 3. Remove background
        resp = await send_action(ws, "BACKGROUND", prompt="white background")
        print("Background response:", resp["message"], resp["status"])
        if "image" in resp:
            display_base64_image(resp["image"])

        # 4. Design garment
        resp = await send_action(ws, "DESIGN", prompt="A white polo shirt with red stripes on the collar and sleeve cuff")

        # 4. Fit garment
        resp = await send_action(ws, "FIT")
        print("Fit response:", resp)
        if "image" in resp:
            display_base64_image(resp["image"])"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uri = f"ws://{args.host}:{args.port}"
    asyncio.run(main(uri))
    time.sleep(25)
    print("run")
    asyncio.run(main(uri))
