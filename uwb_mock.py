import asyncio
import json
import math
import time
import websockets

HOST = "localhost"
PORT = 3000

TAGS = [
    {"deviceId": "2435000088", "x": 2000, "y": 2000},
    {"deviceId": "2435000119", "x": 4000, "y": 4000},
]

INTERVAL = 0.1   # seconds between messages
SPEED    = 30    # cm/step — how fast tags move
RADIUS   = 1500  # cm — 15m circle, clearly visible at scale=1

connected_clients = set()


def simulate_positions(t: float) -> list[dict]:
    now_ms = int(time.time() * 1000)
    messages = []
    for i, tag in enumerate(TAGS):
        angle = t + i * math.pi  # tags move in opposite phase
        cx = tag["x"] + int(RADIUS * math.cos(angle))
        cy = tag["y"] + int(RADIUS * math.sin(angle))

        messages.append({
            "deviceId":     tag["deviceId"],
            "timestamp":    now_ms,
            "twrTimestamp": now_ms - 11,
            "x": cx,
            "y": cy,
            "z": 0,
        })
    return messages


async def broadcast(websocket):
    connected_clients.add(websocket)
    print(f"[Mock UWB] Client connected: {websocket.remote_address}")
    try:
        t = 0.0
        while True:
            for msg in simulate_positions(t):
                await websocket.send(json.dumps(msg))
            t += SPEED * INTERVAL * 0.01   # advance angle
            await asyncio.sleep(INTERVAL)
    except websockets.exceptions.ConnectionClosed:
        print(f"[Mock UWB] Client disconnected: {websocket.remote_address}")
    finally:
        connected_clients.discard(websocket)


async def main():
    print(f"[Mock UWB] Server running on ws://{HOST}:{PORT}/realtime")
    print(f"[Mock UWB] Simulating tags: {[t['deviceId'] for t in TAGS]}")
    async with websockets.serve(broadcast, HOST, PORT):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
