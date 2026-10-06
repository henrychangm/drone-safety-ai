"""Milestone 2: RTSP -> OpenCV -> YOLO (personas) -> ByteTrack (IDs) -> ventana en tiempo real.

Ejecutar desde la raíz del proyecto:  python -m src.main
"""
import argparse
import logging
import threading
import time

import cv2

from src.detection.detector import Detector, PersonDetector
from src.incidents.manager import IncidentManager
from src.safety.ppe import PPE_LABELS, PPEMonitor, State
from src.safety.rules import RuleEngine, RulesConfig
from src.stream.rtsp_reader import RTSPReader
from src.tracking.tracker import PersonTracker
from src.utils.config import load_config

WINDOW = "Drone Safety AI  (Q para salir)"
GREEN = (0, 200, 0)
log = logging.getLogger("main")


RED, YELLOW, BLUE = (0, 0, 230), (0, 200, 230), (230, 140, 0)
SYM = {State.WITH: "OK", State.WITHOUT: "NO", State.UNKNOWN: "?"}


def _label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y, th + 6)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw(frame, people, fps: float, infer_ms: float, ppe_status=None, ppe_dets=(), incident_count=None):
    for d in ppe_dets:  # detecciones PPE crudas, finas, para depurar
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), BLUE, 1)
        _label(frame, f"{d.label} {d.confidence:.2f}", d.x1, d.y1, BLUE)
    for d in people:
        text, color = f"Person #{d.track_id} {d.confidence:.2f}", GREEN
        if ppe_status is not None and d.track_id in ppe_status:
            s = ppe_status[d.track_id]
            text = f"#{d.track_id} H:{SYM[s.helmet.state]} V:{SYM[s.vest.state]}"
            states = (s.helmet.state, s.vest.state)
            color = RED if State.WITHOUT in states else YELLOW if State.UNKNOWN in states else GREEN
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        _label(frame, text, d.x1, d.y1, color)

    hud = f"FPS {fps:4.1f} | infer {infer_ms:4.0f} ms | personas {len(people)}"
    if incident_count is not None:
        hud += f" | incidencias {incident_count}"
    cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def waiting_screen(connected: bool):
    import numpy as np

    img = np.zeros((360, 640, 3), dtype=np.uint8)
    msg = "Esperando frames..." if connected else "Reconectando al stream RTSP..."
    cv2.putText(img, msg, (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)
    return img


def main() -> int:
    parser = argparse.ArgumentParser(description="Drone Safety AI - Milestone 1")
    parser.add_argument("--url", help="Sobrescribe RTSP_URL")
    parser.add_argument("--webcam", nargs="?", const=0, type=int, metavar="INDEX",
                        help="Usa la cámara del Mac (índice 0 por defecto) en vez del RTSP del dron")
    parser.add_argument("--ppe", action="store_true", help="Activa detección de casco/chaleco (requiere modelo PPE)")
    parser.add_argument("--no-window", action="store_true", help="Sin ventana (solo logs)")
    parser.add_argument("--max-seconds", type=float, default=0, help="Salir tras N segundos (pruebas)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    url = args.webcam if args.webcam is not None else (args.url or cfg.rtsp_url)
    show = cfg.show_window and not args.no_window

    log.info("Cargando modelo %s", cfg.model_file)
    detector = PersonDetector(cfg.model_file, cfg.confidence_threshold, cfg.inference_imgsz, cfg.device)
    detector.warmup()

    ppe_detector = ppe_monitor = rules = incidents = None
    n_incidents = 0
    if args.ppe:
        if not cfg.ppe_model_file.exists():
            raise SystemExit(f"No existe el modelo PPE: {cfg.ppe_model_file}\nEntrénalo con: python scripts/train_ppe.py")
        log.info("Cargando modelo PPE %s", cfg.ppe_model_file)
        ppe_detector = Detector(cfg.ppe_model_file, PPE_LABELS, cfg.ppe_confidence, cfg.ppe_imgsz, cfg.device)
        ppe_detector.warmup()
        ppe_monitor = PPEMonitor()
        rules = RuleEngine(RulesConfig(cfg.violation_duration_s, cfg.helmet_inferred_duration_s,
                                       cfg.vest_inferred_duration_s, cfg.incident_cooldown_s,
                                       cfg.require_explicit_helmet))
        incidents = IncidentManager(cfg.data_dir, "webcam" if isinstance(url, int) else str(url),
                                    cfg.snapshot_retention_days)
    tracker = PersonTracker(cfg.tracking_timeout_s, cfg.process_fps_estimate, high_thresh=cfg.confidence_threshold)
    reader = RTSPReader(url, cfg.reconnect_delay_s, cfg.reconnect_max_delay_s, cfg.stall_timeout_s).start()

    stop = threading.Event()
    shared = {"image": None, "error": None}   # el hilo de inferencia publica; la ventana solo lee
    lock = threading.Lock()

    def inference_loop() -> None:
        """Lee frames, detecta, trackea, evalúa PPE y registra incidencias. Nunca toca la ventana."""
        nonlocal n_incidents
        last_id, epoch, fps, infer_ms = -1, -1, 0.0, 0.0
        last_done = last_log = time.monotonic()
        try:
            while not stop.is_set():
                packet = reader.read(last_id, timeout=0.5)
                if packet is None:
                    if incidents:
                        incidents.heartbeat(0.0, 0, reader.connected)
                    continue

                if reader.epoch != epoch:  # stream (re)conectado: IDs antiguos ya no son válidos
                    epoch = reader.epoch
                    tracker.reset()
                    if ppe_monitor:
                        ppe_monitor.reset()
                        rules.reset()
                last_id = packet.frame_id
                t0 = time.monotonic()
                people = tracker.update(detector.detect(packet.frame), packet.frame)
                ppe_dets, ppe_status = [], None
                if ppe_detector:
                    ppe_dets = ppe_detector.detect(packet.frame)
                    ppe_status = ppe_monitor.update(people, ppe_dets, time.monotonic())
                    for v in rules.evaluate(ppe_status, time.monotonic()):
                        evidence = draw(packet.frame.copy(), people, fps, infer_ms, ppe_status, ppe_dets)
                        incidents.create(v, evidence)
                        n_incidents += 1
                infer_ms = (time.monotonic() - t0) * 1000

                now = time.monotonic()
                inst = 1.0 / max(now - last_done, 1e-6)
                fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst  # media móvil exponencial
                last_done = now
                if incidents:
                    incidents.heartbeat(fps, len(people), True)
                if now - last_log > 2:
                    log.info("frame %d | %.1f FPS | %.0f ms | %d personas (IDs %s) | latencia %.0f ms",
                             packet.frame_id, fps, infer_ms, len(people), sorted(p.track_id for p in people),
                             (now - packet.timestamp) * 1000)
                    last_log = now
                img = draw(packet.frame, people, fps, infer_ms, ppe_status, ppe_dets,
                           n_incidents if incidents else None)
                with lock:
                    shared["image"] = img
        except Exception as e:  # se re-lanza en el hilo principal
            log.exception("Fallo en el hilo de inferencia")
            shared["error"] = e
        finally:
            stop.set()

    worker = threading.Thread(target=inference_loop, name="inference", daemon=True)
    worker.start()
    t_start = time.monotonic()
    ui_ticks = 0
    if show:  # ventana redimensionable, con tamaño cómodo para una pantalla de portátil
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 960, 540)
    try:
        # Bucle de interfaz: waitKey ~60 veces/s mantiene viva la ventana (mover, redimensionar, cerrar)
        # aunque la inferencia tarde cientos de ms. En macOS las llamadas de OpenCV GUI van en el hilo principal.
        while not stop.is_set():
            if show:
                with lock:
                    img = shared["image"]
                cv2.imshow(WINDOW, img if img is not None and reader.connected else waiting_screen(reader.connected))
                ui_ticks += 1
                if cv2.waitKey(15) & 0xFF in (ord("q"), ord("Q")):
                    break
            else:
                time.sleep(0.05)
            if args.max_seconds and time.monotonic() - t_start > args.max_seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Cerrando... (interfaz a %.0f Hz)", ui_ticks / max(time.monotonic() - t_start, 1e-6))
        stop.set()
        worker.join(timeout=5)
        reader.stop()
        if incidents:
            incidents.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    if shared["error"]:
        raise shared["error"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
