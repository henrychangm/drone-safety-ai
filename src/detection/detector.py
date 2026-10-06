"""Detector YOLO (Ultralytics) genérico, filtrable por nombre de clase."""
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str  # en minúsculas


class Detector:
    """`labels`: nombres de clase a conservar (None = todas). Se resuelven contra model.names."""

    def __init__(self, model_file: Path, labels: set[str] | None = None, confidence: float = 0.35,
                 imgsz: int = 640, device: str = "cpu"):
        model_file.parent.mkdir(parents=True, exist_ok=True)
        # Si el archivo no existe, Ultralytics descarga el modelo con ese nombre (pesos oficiales).
        self.model = YOLO(str(model_file))
        self.names = {i: n.lower() for i, n in self.model.names.items()}
        if labels is None:
            self.class_ids = None
        else:
            wanted = {l.lower() for l in labels}
            self.class_ids = [i for i, n in self.names.items() if n in wanted]
            missing = wanted - set(self.names.values())
            if missing:
                raise ValueError(f"El modelo {model_file.name} no tiene las clases {sorted(missing)}. "
                                 f"Disponibles: {sorted(self.names.values())}")
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device

    def warmup(self) -> None:
        self.detect(np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            frame, classes=self.class_ids, conf=self.confidence, imgsz=self.imgsz,
            device=self.device, verbose=False,
        )[0]
        out: list[Detection] = []
        for box in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            out.append(Detection(x1, y1, x2, y2, float(box.conf[0]), self.names[int(box.cls[0])]))
        return out


class PersonDetector(Detector):
    """Personas con un modelo COCO (clase 'person')."""

    def __init__(self, model_file: Path, confidence: float = 0.35, imgsz: int = 640, device: str = "cpu"):
        super().__init__(model_file, {"person"}, confidence, imgsz, device)
