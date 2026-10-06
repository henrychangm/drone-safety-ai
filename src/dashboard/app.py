"""Dashboard de revisión humana de incidencias (local).

Ejecutar desde la raíz:  python -m src.dashboard.app   ->  http://127.0.0.1:8000
Solo escucha en localhost: las incidencias pueden incluir nombres de trabajadores (dato personal).
"""
import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.incidents.store import IncidentStore, ReviewError
from src.utils.config import PROJECT_ROOT

STATIC = Path(__file__).parent / "static"


class UpdateBody(BaseModel):
    status: str | None = None
    worker_name: str | None = None
    type: str | None = None
    notes: str | None = None


class MergeBody(BaseModel):
    primary_id: str
    ids: list[str]


def create_app(data_dir: Path) -> FastAPI:
    store = IncidentStore(data_dir / "incidents" / "incidents.db")
    snapshots = data_dir / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Drone Safety Monitor", docs_url=None, redoc_url=None)

    def with_url(row: dict) -> dict:
        p = row.get("snapshot_path")
        # snapshot_path es relativo a data/ ("snapshots/..."); se sirve solo la carpeta snapshots
        row["snapshot_url"] = "/files/" + p.removeprefix("snapshots/") if p else None
        return row

    @app.exception_handler(ReviewError)
    async def review_error(_, exc: ReviewError):
        return JSONResponse({"detail": str(exc)}, status_code=exc.code)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/stats")
    def stats():
        return store.stats()

    @app.get("/api/pipeline")
    def pipeline():
        return store.pipeline()

    @app.get("/api/timeline")
    def timeline(hours: int = Query(24, ge=1, le=168)):
        return store.timeline(hours)

    @app.get("/api/incidents")
    def incidents(status: str | None = None, type: str | None = None, q: str | None = None,
                  limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
        res = store.search(status or None, type or None, q or None, limit, offset)
        res["items"] = [with_url(r) for r in res["items"]]
        return res

    @app.get("/api/incidents/{incident_id}")
    def incident(incident_id: str):
        res = store.get(incident_id)
        res["merged"] = [with_url(m) for m in res["merged"]]
        return with_url(res)

    @app.patch("/api/incidents/{incident_id}")
    def update(incident_id: str, body: UpdateBody):
        res = store.update(incident_id, **body.model_dump(exclude_unset=True))
        res["merged"] = [with_url(m) for m in res["merged"]]
        return with_url(res)

    @app.post("/api/incidents/merge")
    def merge(body: MergeBody):
        res = store.merge(body.primary_id, body.ids)
        res["merged"] = [with_url(m) for m in res["merged"]]
        return with_url(res)

    @app.post("/api/incidents/{incident_id}/unmerge")
    def unmerge(incident_id: str):
        return with_url(store.unmerge(incident_id))

    @app.get("/api/workers")
    def workers():
        return store.workers()

    app.mount("/files", StaticFiles(directory=snapshots), name="files")
    app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Dashboard de incidencias")
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    uvicorn.run(create_app(Path(a.data_dir)), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    print("Dashboard en http://127.0.0.1:8000  (Ctrl+C para salir)")
    main()
