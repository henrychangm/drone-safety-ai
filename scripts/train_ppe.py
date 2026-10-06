"""Fine-tuning de yolo11n sobre construction-ppe. Ejecutar desde la raíz:  python scripts/train_ppe.py"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", default="ppe-v1")
    ap.add_argument("--fraction", type=float, default=1.0, help="Fracción del dataset (pruebas rápidas)")
    a = ap.parse_args()

    model = YOLO(str(ROOT / "models" / "yolo11n.pt"))
    model.train(
        data=str(ROOT / "data" / "construction-ppe.yaml"),
        epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=4,
        fraction=a.fraction, project=str(ROOT / "models" / "runs"), name=a.name,
        patience=10, plots=False,
    )
