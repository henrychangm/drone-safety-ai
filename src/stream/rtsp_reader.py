"""Lectura RTSP (o webcam, si `url` es un int) en hilo propio que conserva SOLO el último frame.

Separa captura de inferencia: si la inferencia tarda más que el intervalo entre frames,
los frames antiguos se descartan en lugar de acumularse (latencia acotada).
"""
import logging
import os
import threading
import time
from dataclasses import dataclass

# Debe fijarse antes de que OpenCV abra el primer stream.
# TCP evita artefactos por pérdida UDP; nobuffer/low_delay reducen latencia.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay",
)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FramePacket:
    frame: np.ndarray
    frame_id: int      # contador creciente de frames capturados
    timestamp: float   # time.monotonic() en el momento de captura


class RTSPReader:
    def __init__(
        self,
        url: str | int,
        reconnect_delay_s: float = 1.0,
        reconnect_max_delay_s: float = 10.0,
        stall_timeout_s: float = 5.0,
    ):
        self.url = url
        self.reconnect_delay_s = reconnect_delay_s
        self.reconnect_max_delay_s = reconnect_max_delay_s
        self.stall_timeout_s = stall_timeout_s

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest: FramePacket | None = None
        self._frame_id = 0
        self._connected = False
        self._epoch = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rtsp-reader", daemon=True)

    # --- API pública -------------------------------------------------
    def start(self) -> "RTSPReader":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=5)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def epoch(self) -> int:
        """Aumenta en cada (re)conexión; permite resetear estado dependiente del stream."""
        return self._epoch

    def read(self, last_id: int = -1, timeout: float = 1.0) -> FramePacket | None:
        """Devuelve el último frame con id > last_id, o None si expira el timeout."""
        with self._cond:
            self._cond.wait_for(
                lambda: self._stop.is_set()
                or (self._latest is not None and self._latest.frame_id > last_id),
                timeout=timeout,
            )
            if self._latest is not None and self._latest.frame_id > last_id:
                return self._latest
            return None

    # --- Hilo de captura ---------------------------------------------
    def _open(self) -> cv2.VideoCapture | None:
        if isinstance(self.url, int):  # webcam local (macOS: AVFoundation)
            cap = cv2.VideoCapture(self.url, cv2.CAP_AVFOUNDATION)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
        return None

    def _run(self) -> None:
        delay = self.reconnect_delay_s
        while not self._stop.is_set():
            log.info("Conectando a %s ...", self.url)
            cap = self._open()
            if cap is None:
                log.warning("No se pudo abrir el stream. Reintento en %.1fs", delay)
                self._stop.wait(delay)
                delay = min(delay * 2, self.reconnect_max_delay_s)
                continue

            log.info("Stream conectado")
            self._epoch += 1
            self._connected = True
            delay = self.reconnect_delay_s
            last_ok = time.monotonic()

            while not self._stop.is_set():
                ok, frame = cap.read()
                now = time.monotonic()
                if ok and frame is not None:
                    last_ok = now
                    with self._cond:
                        self._frame_id += 1
                        self._latest = FramePacket(frame, self._frame_id, now)
                        self._cond.notify_all()
                elif now - last_ok > self.stall_timeout_s or not cap.isOpened():
                    log.warning("Stream perdido (sin frames %.1fs)", now - last_ok)
                    break
                else:
                    time.sleep(0.01)

            self._connected = False
            cap.release()
            self._stop.wait(delay)
        self._connected = False
