import threading
import time

import cv2


class ThreadedStreamCapture:

    def __init__(self, url, reconnect_delay=2.0):
        self.url = url
        self.reconnect_delay = reconnect_delay

        self._cap = cv2.VideoCapture(url)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._running = True
        self._opened = self._cap.isOpened()

        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

        start = time.time()
        while time.time() - start < 5:
            with self._frame_lock:
                if self._latest_frame is not None:
                    return
            time.sleep(0.05)

    def _update_loop(self):
        while self._running:
            if not self._cap.isOpened():
                self._opened = False
                time.sleep(self.reconnect_delay)
                self._cap.open(self.url)
                continue

            ret, frame = self._cap.read()
            if ret:
                self._opened = True
                with self._frame_lock:
                    self._latest_frame = frame
            else:
                self._opened = False
                self._cap.release()
                time.sleep(self.reconnect_delay)
                self._cap = cv2.VideoCapture(self.url)

    def isOpened(self):
        return self._opened

    def read(self):
        with self._frame_lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def release(self):
        self._running = False
        self._cap.release()