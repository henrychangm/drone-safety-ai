"""Acceso a SQLite de incidencias, compartido por el pipeline (escribe) y el dashboard (revisa).

Una conexión por operación + WAL: dos procesos (pipeline y dashboard) pueden usar la misma BD.
Flujo de revisión humana: pending -> confirmed | discarded; las repetidas se fusionan (merged).
"""
import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

EDITABLE_STATUSES = ("pending", "confirmed", "discarded")
TYPES = ("no_helmet", "no_vest", "no_helmet,no_vest")   # el último: faltan ambos EPI

SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id            TEXT PRIMARY KEY,
    timestamp     TEXT NOT NULL,      -- UTC ISO-8601
    track_id      INTEGER NOT NULL,   -- ID temporal del tracker; NO identifica a una persona real
    type          TEXT NOT NULL,      -- no_helmet | no_vest | no_helmet,no_vest
    confidence    REAL NOT NULL,
    duration_s    REAL NOT NULL,
    explicit      INTEGER NOT NULL,   -- 1 = detectado por clase no_*, 0 = inferido por ausencia
    snapshot_path TEXT,
    source        TEXT
);
CREATE INDEX IF NOT EXISTS idx_incidents_ts ON incidents(timestamp);
CREATE TABLE IF NOT EXISTS review_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id TEXT NOT NULL,
    ts          TEXT NOT NULL,
    action      TEXT NOT NULL,
    detail      TEXT
);
CREATE INDEX IF NOT EXISTS idx_log_incident ON review_log(incident_id);
CREATE TABLE IF NOT EXISTS pipeline_status (   -- latido del pipeline de visión (una fila)
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    session_id TEXT,
    source     TEXT,
    updated_at REAL,      -- epoch; 0 = detenido limpiamente
    started_at REAL,
    fps        REAL,
    people     INTEGER,
    connected  INTEGER    -- 1 = llegan frames
);
"""

# Columnas añadidas después de la v1: se migran sin perder datos existentes.
EXTRA_COLUMNS = {
    "status": "TEXT NOT NULL DEFAULT 'pending'",   # pending | confirmed | discarded | merged
    "worker_name": "TEXT",                         # asignado a mano por el revisor
    "merged_into": "TEXT",                         # id de la incidencia principal si status = merged
    "reviewed_at": "TEXT",
    "notes": "TEXT",
    "session_id": "TEXT",                          # una ejecución del pipeline / recorrido
}


class ReviewError(Exception):
    def __init__(self, message: str, code: int = 400):
        super().__init__(message)
        self.code = code


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class IncidentStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)
            have = {r["name"] for r in c.execute("PRAGMA table_info(incidents)")}
            for col, ddl in EXTRA_COLUMNS.items():
                if col not in have:
                    c.execute(f"ALTER TABLE incidents ADD COLUMN {col} {ddl}")
            c.execute("CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status)")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _log(c, incident_id: str, action: str, detail: dict | None = None) -> None:
        c.execute("INSERT INTO review_log (incident_id, ts, action, detail) VALUES (?,?,?,?)",
                  (incident_id, _now(), action, json.dumps(detail, ensure_ascii=False) if detail else None))

    # --- Escritura desde el pipeline ----------------------------------
    def insert(self, *, id: str, timestamp: str, track_id: int, type: str, confidence: float,
               duration_s: float, explicit: bool, snapshot_path: str | None, source: str,
               session_id: str) -> None:
        with self._connect() as c:
            c.execute(
                "INSERT INTO incidents (id,timestamp,track_id,type,confidence,duration_s,explicit,"
                "snapshot_path,source,session_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (id, timestamp, track_id, type, confidence, duration_s, int(explicit),
                 snapshot_path, source, session_id))

    def count(self) -> int:
        with self._connect() as c:
            return c.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]

    # --- Latido del pipeline ------------------------------------------
    def heartbeat(self, session_id: str, source: str, fps: float, people: int, connected: bool,
                  started_at: float) -> None:
        with self._connect() as c:
            c.execute("INSERT OR REPLACE INTO pipeline_status VALUES (1,?,?,?,?,?,?,?)",
                      (session_id, source, time.time(), started_at, fps, people, int(connected)))

    def stop_heartbeat(self, session_id: str) -> None:
        with self._connect() as c:
            c.execute("UPDATE pipeline_status SET updated_at = 0 WHERE id = 1 AND session_id = ?", (session_id,))

    def pipeline(self, stale_after_s: float = 10.0) -> dict:
        with self._connect() as c:
            r = c.execute("SELECT * FROM pipeline_status WHERE id = 1").fetchone()
        if r is None:
            return {"active": False, "last_seen_s": None}
        age = time.time() - r["updated_at"] if r["updated_at"] else None
        active = age is not None and age < stale_after_s
        return {"active": active, "last_seen_s": age if r["updated_at"] else None, "source": r["source"],
                "session_id": r["session_id"], "fps": r["fps"] if active else 0, "people": r["people"] if active else 0,
                "connected": bool(r["connected"]) if active else False,
                "uptime_s": time.time() - r["started_at"] if active else None}

    def timeline(self, hours: int = 24) -> list[dict]:
        """Incidencias por hora (UTC, hora completa) de las últimas `hours`, sin descartadas ni repeticiones."""
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start = end - timedelta(hours=hours - 1)
        buckets = {(start + timedelta(hours=k)).isoformat(timespec="seconds"): {t: 0 for t in TYPES}
                   for k in range(hours)}
        with self._connect() as c:
            rows = c.execute("SELECT timestamp, type FROM incidents WHERE timestamp >= ? "
                             "AND status IN ('pending','confirmed')", (start.isoformat(timespec="seconds"),))
            for r in rows:
                ts = datetime.fromisoformat(r["timestamp"]).astimezone(timezone.utc)
                key = ts.replace(minute=0, second=0, microsecond=0).isoformat(timespec="seconds")
                if key in buckets and r["type"] in buckets[key]:
                    buckets[key][r["type"]] += 1
        return [{"hour": k, "no_helmet": v["no_helmet"], "no_vest": v["no_vest"], "both": v["no_helmet,no_vest"]}
                for k, v in buckets.items()]

    # --- Lectura -------------------------------------------------------
    def stats(self) -> dict:
        with self._connect() as c:
            by_status = {s: 0 for s in (*EDITABLE_STATUSES, "merged")}
            for r in c.execute("SELECT status, COUNT(*) n FROM incidents GROUP BY status"):
                by_status[r["status"]] = r["n"]
            by_type = {t: 0 for t in TYPES}
            for r in c.execute("SELECT type, COUNT(*) n FROM incidents "
                               "WHERE status IN ('pending','confirmed') GROUP BY type"):
                by_type[r["type"]] = r["n"]
        return {
            "total": by_status["pending"] + by_status["confirmed"] + by_status["discarded"],
            "pending": by_status["pending"], "confirmed": by_status["confirmed"],
            "discarded": by_status["discarded"], "merged_repeats": by_status["merged"],
            "by_type": by_type,
        }

    def search(self, status: str | None = None, type: str | None = None, q: str | None = None,
             limit: int = 100, offset: int = 0) -> dict:
        where, args = [], []
        if status == "merged":
            where.append("i.status = 'merged'")
        elif status:
            where.append("i.status = ?")
            args.append(status)
        else:
            where.append("i.status != 'merged'")
        if type:  # "no_helmet" incluye también los casos en que faltan ambos EPI
            where.append("(',' || i.type || ',') LIKE ?")
            args.append(f"%,{type},%")
        if q:
            where.append("(i.worker_name LIKE ? OR i.notes LIKE ? OR i.id LIKE ?)")
            args += [f"%{q}%"] * 3
        clause = " AND ".join(where)
        with self._connect() as c:
            total = c.execute(f"SELECT COUNT(*) FROM incidents i WHERE {clause}", args).fetchone()[0]
            rows = c.execute(
                f"SELECT i.*, (SELECT COUNT(*) FROM incidents m WHERE m.merged_into = i.id) AS merged_count "
                f"FROM incidents i WHERE {clause} ORDER BY i.timestamp DESC LIMIT ? OFFSET ?",
                [*args, limit, offset]).fetchall()
        return {"total": total, "items": [dict(r) for r in rows]}

    def get(self, incident_id: str) -> dict:
        with self._connect() as c:
            row = c.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
            if row is None:
                raise ReviewError("Incidencia no encontrada", 404)
            merged = c.execute("SELECT * FROM incidents WHERE merged_into = ? ORDER BY timestamp",
                               (incident_id,)).fetchall()
            log = c.execute("SELECT ts, action, detail FROM review_log WHERE incident_id = ? "
                            "ORDER BY id DESC LIMIT 50", (incident_id,)).fetchall()
        return {**dict(row), "merged": [dict(m) for m in merged], "log": [dict(l) for l in log]}

    def workers(self) -> list[str]:
        with self._connect() as c:
            return [r[0] for r in c.execute(
                "SELECT DISTINCT worker_name FROM incidents WHERE worker_name IS NOT NULL "
                "AND worker_name != '' ORDER BY worker_name COLLATE NOCASE")]

    # --- Revisión humana -----------------------------------------------
    def update(self, incident_id: str, *, status: str | None = None, worker_name: str | None = None,
               type: str | None = None, notes: str | None = None) -> dict:
        changes: dict = {}
        if status is not None:
            if status not in EDITABLE_STATUSES:
                raise ReviewError(f"Estado no válido: {status}")
            changes["status"] = status
        if type is not None:
            if type not in TYPES:
                raise ReviewError(f"Tipo no válido: {type}")
            changes["type"] = type
        if worker_name is not None:
            changes["worker_name"] = worker_name.strip() or None
        if notes is not None:
            changes["notes"] = notes.strip() or None
        if not changes:
            raise ReviewError("Nada que actualizar")

        with self._connect() as c:
            row = c.execute("SELECT status FROM incidents WHERE id = ?", (incident_id,)).fetchone()
            if row is None:
                raise ReviewError("Incidencia no encontrada", 404)
            if row["status"] == "merged":
                raise ReviewError("Está fusionada: sepárala o edita la incidencia principal", 409)
            sets = ", ".join(f"{k} = ?" for k in changes)
            c.execute(f"UPDATE incidents SET {sets}, reviewed_at = ? WHERE id = ?",
                      [*changes.values(), _now(), incident_id])
            self._log(c, incident_id, "update", changes)
        return self.get(incident_id)

    def merge(self, primary_id: str, ids: list[str]) -> dict:
        others = [i for i in dict.fromkeys(ids) if i != primary_id]
        if not others:
            raise ReviewError("Selecciona al menos otra incidencia además de la principal")
        with self._connect() as c:
            primary = c.execute("SELECT status FROM incidents WHERE id = ?",
                                (primary_id,)).fetchone()
            if primary is None:
                raise ReviewError("Incidencia principal no encontrada", 404)
            if primary["status"] == "merged":
                raise ReviewError("La principal ya está fusionada en otra", 409)
            for oid in others:
                row = c.execute("SELECT id FROM incidents WHERE id = ?", (oid,)).fetchone()
                if row is None:
                    raise ReviewError(f"Incidencia {oid} no encontrada", 404)
                c.execute("UPDATE incidents SET merged_into = ? WHERE merged_into = ?", (primary_id, oid))
                # No se toca el resto de campos de la repetición: así "Separar" no pierde datos.
                c.execute("UPDATE incidents SET status='merged', merged_into=?, reviewed_at=? WHERE id = ?",
                          (primary_id, _now(), oid))
                self._log(c, oid, "merged_into", {"primary": primary_id})
            self._log(c, primary_id, "merge", {"absorbed": others})
        return self.get(primary_id)

    def unmerge(self, incident_id: str) -> dict:
        with self._connect() as c:
            row = c.execute("SELECT status, merged_into FROM incidents WHERE id = ?", (incident_id,)).fetchone()
            if row is None:
                raise ReviewError("Incidencia no encontrada", 404)
            if row["status"] != "merged":
                raise ReviewError("No está fusionada", 409)
            c.execute("UPDATE incidents SET status='pending', merged_into=NULL, reviewed_at=? WHERE id = ?",
                      (_now(), incident_id))
            self._log(c, incident_id, "unmerged", {"from": row["merged_into"]})
        return self.get(incident_id)
