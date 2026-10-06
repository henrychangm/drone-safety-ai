# Drone Safety AI

Pipeline: DJI Mini 5 Pro → RC 2 → RTMP → MediaMTX → `rtsp://127.0.0.1:8554/drone` → OpenCV → YOLO.

## Setup (Mac Intel)
```bash
/usr/local/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # opcional
```

## Milestones 1-2 – personas con ID de tracking (ByteTrack)
1. `mediamtx ~/mediamtx.yml`
2. Iniciar livestream en DJI Fly.
3. `python -m src.main` (Q para salir). Sin dron: `python -m src.main --webcam`

Opciones: `--url rtsp://...`, `--no-window`, `--max-seconds N`.

## Dashboard de revisión humana
```bash
python -m src.dashboard.app            # http://127.0.0.1:8000 (solo localhost)
```
El pipeline (`python -m src.main --ppe`) guarda incidencias; en el dashboard una persona las revisa:
aprobar / descartar, asignar el nombre del trabajador, corregir el tipo, fusionar repeticiones
(y separarlas) y ver la evidencia. Cada cambio queda en la tabla `review_log`.
Sin dron ni cámara: `python scripts/seed_demo_incidents.py --data-dir data_demo` y
`python -m src.dashboard.app --data-dir data_demo`.

## Dashboard siempre encendido (macOS)
```bash
scripts/dashboard_service.sh install     # arranca al iniciar sesión y se reinicia si se cae
scripts/dashboard_service.sh status | restart | logs | uninstall
```
El proyecto vive en `~/drone-safety-ai` (no en el Escritorio): macOS bloquea a los procesos en segundo plano
que intentan leer Escritorio/Documentos sin permiso. Las versiones exactas de dependencias están en `requirements-lock.txt`.
