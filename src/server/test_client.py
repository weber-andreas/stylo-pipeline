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
import time

async def send_action(ws, action, **kwargs):
    payload = {"action": action, **kwargs}
    await ws.send(json.dumps(payload))
    response = await ws.recv()
    return json.loads(response)

def display_base64_image(b64_str):
    img_data = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_data))
    img.save("src/server/example_transfers.png")

async def main(uri):
    async with websockets.connect(uri, max_size=10 * 1024 * 1024) as ws:
        """async for message in ws:
            print("Received message:", message)
            #close connection if server sends a close message
            if isinstance(message, str) and message.startswith("ERROR: Another client is already connected. Try again later."):
                print("Server requested close:", message)
                await ws.close()
                return"""
        print("Connected to server.")
        time.sleep(50)

        # 1. List available commands
        resp = await send_action(ws, "LIST")
        print("Server LIST Response:", resp)

        # 2. Upload an image
        with open("results/stable_viton_output.png", "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")
        resp = await send_action(ws, "UPLOAD", image=b64_img)
        print("Upload response:", resp)

        # 3. Remove background
        resp = await send_action(ws, "BACKGROUND", prompt="white background")
        print("Background response:", resp["message"], resp["status"])
        if "image" in resp:
            display_base64_image(resp["image"])

        # 4. Fit garment
        resp = await send_action(ws, "FIT")
        print("Fit response:", resp)
        #if "image" in resp:
        #    display_base64_image(resp["image"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uri = f"ws://{args.host}:{args.port}"
    asyncio.run(main(uri))
