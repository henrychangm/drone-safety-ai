"""Tracking de personas con ByteTrack (implementación de Ultralytics).

Desacoplado del detector: recibe `Detection`s y devuelve `TrackedPerson`s con un ID
temporal (no biométrico). Los IDs solo identifican a alguien mientras el tracker lo sigue.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from src.detection.detector import Detection


@dataclass(frozen=True)
class TrackedPerson:
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class _Boxes:
    """Mínimo que BYTETracker espera: conf/xywh/cls con indexado booleano."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh, self.conf, self.cls = xywh, conf, cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "_Boxes":
        return _Boxes(self.xywh[idx], self.conf[idx], self.cls[idx])


class PersonTracker:
    def __init__(
        self,
        tracking_timeout_s: float = 3.0,
        process_fps: float = 10.0,
        high_thresh: float = 0.35,
        low_thresh: float = 0.1,
        new_track_thresh: float = 0.4,
        match_thresh: float = 0.8,
    ):
        # ByteTrack cuenta en frames *procesados*, no en segundos: convertimos el timeout.
        self._args = SimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=high_thresh,
            track_low_thresh=low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=max(1, int(tracking_timeout_s * process_fps)),
            match_thresh=match_thresh,
            fuse_score=True,
        )
        self._tracker = BYTETracker(self._args)

    def reset(self) -> None:
        """Descarta todos los tracks (p. ej. tras reconectar el stream)."""
        self._tracker = BYTETracker(self._args)

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[TrackedPerson]:
        if detections:
            xyxy = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32)
            xywh = np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2,
                                    xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]])
            boxes = _Boxes(xywh, np.array([d.confidence for d in detections], dtype=np.float32),
                           np.zeros(len(detections), dtype=np.float32))
        else:
            boxes = _Boxes(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32))

        rows = self._tracker.update(boxes, frame)
        # Fila: [x1, y1, x2, y2, track_id, score, cls, idx]
        return [TrackedPerson(int(r[4]), int(r[0]), int(r[1]), int(r[2]), int(r[3]), float(r[5])) for r in rows]
