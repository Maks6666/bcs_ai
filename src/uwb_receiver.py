import socket
import threading
import numpy as np
import websocket
import json
from collections import defaultdict, deque


class NoccelaPositionManager:
    def __init__(self, url: str, cameras: list, smoothing_window: int = 10):
        self.url = url
        self.cameras = {cam.camera_id: cam for cam in cameras}

        # Последние N сырых позиций по каждому тегу — для сглаживания
        self.position_history = defaultdict(lambda: deque(maxlen=smoothing_window))

    def start(self):
        t = threading.Thread(target=self._connect, daemon=True)
        t.start()

    def _connect(self):
        ws = websocket.WebSocketApp(
            self.url,
            on_message=self._on_message,
            on_error=lambda ws, e: print(f"[Noccela] Ошибка: {e}"),
            on_close=lambda ws, c, m: print("[Noccela] Соединение закрыто")
        )
        ws.run_forever()

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            # here we receive tag_id
            tag_id = data.get("deviceId")

            X_raw = float(data.get("x", 0)) / 20.0
            Z_raw = float(data.get("y", 0)) / 20.0
            height_raw = float(data.get("z", 0)) / 20.0

            
            history = self.position_history[tag_id]
            history.append((X_raw, Z_raw, height_raw))

            n = len(history)
            X = sum(p[0] for p in history) / n
            Z = sum(p[1] for p in history) / n
            height = sum(p[2] for p in history) / n

            cam = self.cameras.get(tag_id)
            if cam:
                cam.update_from_uwb(X, Z, height=height)
                print(f"[Noccela] Тег {tag_id} -> X={X:.2f}, Z={Z:.2f}, height={height:.2f} (avg of {n})")
            else:
                print(f"[Noccela] Unkown tag_id: {tag_id}")
        except Exception as e:
            print(f"[Noccela] Parsing error: {e}")