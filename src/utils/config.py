"""Configuración central. Valores por defecto sobrescribibles vía .env o variables de entorno."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    rtsp_url: str = _env("RTSP_URL", "rtsp://127.0.0.1:8554/drone")

    model_path: str = _env("MODEL_PATH", "models/yolo11n.pt")
    confidence_threshold: float = float(_env("CONFIDENCE_THRESHOLD", "0.35"))
    inference_imgsz: int = int(_env("INFERENCE_IMGSZ", "640"))
    device: str = _env("DEVICE", "cpu")

    reconnect_delay_s: float = float(_env("RECONNECT_DELAY_S", "1.0"))
    reconnect_max_delay_s: float = float(_env("RECONNECT_MAX_DELAY_S", "10.0"))
    stall_timeout_s: float = float(_env("STALL_TIMEOUT_S", "5.0"))

    # Tracking. tracking_timeout_s: cuánto se recuerda a una persona que deja de verse
    tracking_timeout_s: float = float(_env("TRACKING_TIMEOUT_S", "3.0"))
    process_fps_estimate: float = float(_env("PROCESS_FPS_ESTIMATE", "10"))

    # PPE (casco/chaleco)
    ppe_model_path: str = _env("PPE_MODEL_PATH", "models/runs/ppe-v1/weights/best.pt")
    ppe_confidence: float = float(_env("PPE_CONFIDENCE", "0.25"))
    ppe_imgsz: int = int(_env("PPE_IMGSZ", "512"))

    # Reglas de incidencia
    violation_duration_s: float = float(_env("VIOLATION_DURATION_S", "5.0"))
    helmet_inferred_duration_s: float = float(_env("HELMET_INFERRED_DURATION_S", "8.0"))
    vest_inferred_duration_s: float = float(_env("VEST_INFERRED_DURATION_S", "10.0"))
    incident_cooldown_s: float = float(_env("INCIDENT_COOLDOWN_S", "60.0"))
    require_explicit_helmet: bool = _bool("REQUIRE_EXPLICIT_HELMET", False)
    snapshot_retention_days: int = int(_env("SNAPSHOT_RETENTION_DAYS", "30"))

    show_window: bool = _bool("SHOW_WINDOW", True)

    @property
    def ppe_model_file(self) -> Path:
        p = Path(self.ppe_model_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def model_file(self) -> Path:
        p = Path(self.model_path)
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_config() -> Config:
    return Config()
