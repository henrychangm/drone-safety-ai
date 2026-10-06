"""Registro de incidencias: SQLite + snapshot JPEG. Sin vídeo continuo: solo evidencia puntual."""
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.incidents.store import IncidentStore
from src.safety.rules import Violation

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class Incident:
    id: str
    timestamp: str
    violation: Violation
    snapshot_path: str | None


class IncidentManager:
    def __init__(self, data_dir: Path, source: str = "", retention_days: int = 30):
        self.snapshots_dir = data_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "incidents").mkdir(parents=True, exist_ok=True)
        self.source = source
        self.store = IncidentStore(data_dir / "incidents" / "incidents.db")
        self.session_id = uuid.uuid4().hex[:8]  # una ejecución del pipeline = un recorrido
        self._started = time.time()
        self._last_hb = 0.0
        self._purge_old_snapshots(retention_days)

    def create(self, v: Violation, frame: np.ndarray) -> Incident:
        iid = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc)
        ts = now.isoformat(timespec="seconds")
        day_dir = self.snapshots_dir / now.strftime("%Y-%m-%d")
        day_dir.mkdir(exist_ok=True)
        snap = day_dir / f"{now.strftime('%H%M%S')}_{v.type.replace(',', '+')}_p{v.track_id}_{iid}.jpg"
        cv2.imwrite(str(snap), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        rel = str(snap.relative_to(self.snapshots_dir.parent))
        self.store.insert(id=iid, timestamp=ts, track_id=v.track_id, type=v.type, confidence=v.confidence,
                          duration_s=v.duration_s, explicit=v.explicit, snapshot_path=rel,
                          source=self.source, session_id=self.session_id)
        log.warning("INCIDENCIA %s | Person #%d | %s | %.1fs | conf %.2f | %s",
                    iid, v.track_id, v.type, v.duration_s, v.confidence, rel)
        return Incident(iid, ts, v, rel)

    def heartbeat(self, fps: float, people: int, connected: bool) -> None:
        """Avisa al dashboard de que el pipeline sigue vivo (como mucho una vez cada 2 s)."""
        now = time.monotonic()
        if now - self._last_hb >= 2.0:
            self._last_hb = now
            self.store.heartbeat(self.session_id, self.source, fps, people, connected, self._started)

    def count(self) -> int:
        return self.store.count()

    def _purge_old_snapshots(self, days: int) -> None:
        if days <= 0:
            return
        cutoff = time.time() - days * 86400
        removed = 0
        for f in self.snapshots_dir.rglob("*.jpg"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        if removed:
            log.info("Retención: %d snapshots antiguos eliminados", removed)

    def close(self) -> None:
        self.store.stop_heartbeat(self.session_id)  # el dashboard muestra "detención" de inmediato
