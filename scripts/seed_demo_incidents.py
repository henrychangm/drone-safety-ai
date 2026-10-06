"""Genera incidencias FALSAS con snapshots sintéticos para probar el dashboard sin dron.

    python scripts/seed_demo_incidents.py --data-dir data_demo
    python -m src.dashboard.app --data-dir data_demo
"""
import argparse
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.incidents.store import IncidentStore  # noqa: E402


def fake_snapshot(path: Path, track_id: int, kind: str) -> None:
    img = np.full((540, 960, 3), (70, 80, 90), np.uint8)
    cv2.rectangle(img, (0, 400), (960, 540), (60, 110, 120), -1)
    x = random.randint(150, 700)
    cv2.rectangle(img, (x, 120), (x + 110, 440), (0, 0, 230), 3)
    cv2.putText(img, f"#{track_id} " + " ".join(x for x, k in (("H:NO", "no_helmet"), ("V:NO", "no_vest")) if k in kind), (x, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 230), 2)
    cv2.putText(img, "DEMO - imagen sintetica", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_demo")
    ap.add_argument("--n", type=int, default=26)
    a = ap.parse_args()
    data = Path(a.data_dir)
    store = IncidentStore(data / "incidents" / "incidents.db")
    now = datetime.now(timezone.utc)
    for k in range(a.n):
        ts = now - timedelta(minutes=random.randint(1, 1300))
        kind = random.choice(["no_helmet", "no_vest", "no_helmet,no_vest"])
        iid, tid = uuid.uuid4().hex[:12], random.randint(1, 30)
        rel = f"snapshots/{ts:%Y-%m-%d}/{ts:%H%M%S}_{kind.replace(',', '+')}_p{tid}_{iid}.jpg"
        fake_snapshot(data / rel, tid, kind)
        store.insert(id=iid, timestamp=ts.isoformat(timespec="seconds"), track_id=tid, type=kind,
                     confidence=round(random.uniform(0.4, 0.9), 2), duration_s=round(random.uniform(5, 14), 1),
                     explicit="no_helmet" in kind and random.random() < 0.5, snapshot_path=rel, source="demo", session_id="demo")
    print(f"{a.n} incidencias de demo en {data}")
