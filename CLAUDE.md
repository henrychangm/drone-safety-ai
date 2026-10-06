# Drone Safety AI — contexto del proyecto

Supervisión de seguridad en obra con vídeo de un DJI Mini 5 Pro (mando RC 2). Detecta personas, casco y chaleco,
genera incidencias con foto y las revisa un humano en un dashboard. El usuario habla español.

## Cadena de vídeo (ya funciona; no reconstruir)
DJI Fly (RTMP) → MediaMTX en el Mac (`mediamtx ~/mediamtx.yml`, path `drone`) → `rtsp://127.0.0.1:8554/drone` → Python.
No tocar la config de MediaMTX salvo problema real. La URL usa loopback: no depende de la IP del router.
Sin dron: `--webcam` usa la cámara del Mac.

## Entorno
- Mac Intel (x86_64). PyTorch solo publica ruedas hasta 2.2.2 (exige numpy<2) → **Python 3.12**, versiones exactas en
  `requirements-lock.txt`. Entorno: `.venv` (no se puede mover; se recrea con el lock).
- **El proyecto debe vivir fuera de Escritorio/Documentos**: macOS bloquea a los procesos de launchd que leen esas
  carpetas, y el servicio del dashboard se queda colgado. Ubicación actual: `~/drone-safety-ai`.

## Comandos
- Cámara + PPE: `python -m src.main --webcam --ppe` (sin `--webcam` = dron). Q cierra. `--no-window` para pruebas.
- Dashboard (siempre encendido, http://127.0.0.1:8000): `scripts/dashboard_service.sh install|status|restart|logs|uninstall`.
  Manual: `python -m src.dashboard.app [--data-dir data_demo]`. Datos falsos: `scripts/seed_demo_incidents.py`.
- Entrenar PPE: `python scripts/train_ppe.py` (CPU: ~5 min/época; 25 épocas ≈ 3 h).

## Arquitectura (src/)
`stream/rtsp_reader.py` hilo que conserva solo el último frame + reconexión · `detection/detector.py` YOLO genérico ·
`tracking/tracker.py` ByteTrack (IDs temporales, se reinician al reconectar) · `safety/ppe.py` asocia EPI a personas y filtra
en el tiempo (con/sin/desconocido) · `safety/rules.py` duración mínima + cooldown → incidencia ·
`incidents/store.py` SQLite compartido (pipeline escribe, dashboard revisa; WAL) · `incidents/manager.py` snapshot + latido ·
`dashboard/` FastAPI + `static/index.html`. `main.py`: hilo de inferencia aparte; la ventana OpenCV va en el hilo principal
(si no, no se puede arrastrar la ventana).

## Decisiones que no se deducen del código
- "No se detectó casco" ≠ "sin casco": solo cuenta si la zona es visible (persona grande, sin oclusión, cabeza no cortada).
  Evidencia explícita (`no_helmet`) pesa más que la inferida. El dataset NO tiene `no_vest`, el chaleco siempre es inferido.
- Una incidencia por persona y momento: si faltan ambos EPI → tipo `no_helmet,no_vest`. Cooldown por (track, EPI).
- Modelo PPE (`models/runs/ppe-v1/weights/best.pt`): dataset `construction-ppe` de Ultralytics (1.416 imágenes, AGPL-3.0),
  mAP50 ≈ 0,57. helmet/vest ≈ 0,83; `no_helmet` solo recall 0,31. Es válido para pruebas, no para producción.
- Privacidad: IDs del tracker son temporales; **sin reconocimiento facial** (dato biométrico, RGPD). El nombre del trabajador
  lo asigna un humano en el dashboard; un aviso del sistema no es una sanción. Snapshots se purgan a los 30 días.
- Ultralytics `datasets_dir` (config global) apunta a la raíz del proyecto; el yaml usa `path: data/datasets`.
- Paleta del gráfico validada con el validador de dataviz (azul casco, naranja chaleco, aqua ambos).

## Pendiente / próximos pasos
1. Probar con el dron real (vista aérea degrada el modelo). 2. Recoger y etiquetar frames propios y reentrenar.
3. Ajustar `HELMET_INFERRED_DURATION_S` / `VEST_INFERRED_DURATION_S` según falsos positivos. 4. Vídeo en directo en el dashboard.
5. Más adelante: reconocimiento de caras (requiere dataset y revisión legal), vuelo autónomo (problema separado).
