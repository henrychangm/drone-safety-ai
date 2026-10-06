"""Lógica PPE: relaciona cascos/chalecos con personas trackeadas y decide con/sin/desconocido.

Dos capas:
  1. PPEAssociator  -> observación por frame y persona (¿vemos casco? ¿vemos que falta? ¿no se puede saber?)
  2. PPEMonitor     -> filtro temporal por track: estado estable + cuánto lleva sin el EPI.

Principio: "no se detectó casco" NO es "sin casco". Solo es evidencia débil, y solo si la zona
(cabeza/torso) es visible. Evidencia explícita (clase no_helmet) pesa más que la inferida.
Esta capa NO crea incidencias: solo expone estado y duración. Las reglas van en safety/rules.py.
"""
from collections import deque
from dataclasses import dataclass
from enum import Enum

from src.detection.detector import Detection
from src.tracking.tracker import TrackedPerson

ITEMS = ("helmet", "vest")
# Clase del modelo que indica EPI puesto / ausencia explícita (el dataset no tiene no_vest)
WITH_LABEL = {"helmet": "helmet", "vest": "vest"}
WITHOUT_LABEL = {"helmet": "no_helmet", "vest": "no_vest"}
# Clases que se piden al modelo. construction-ppe no tiene no_vest; si un modelo futuro la incluye, añádela.
PPE_LABELS = {"helmet", "vest", "no_helmet"}


class State(Enum):
    WITH = "with"
    WITHOUT = "without"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Observation:
    state: State
    confidence: float = 0.0
    explicit: bool = False  # True si WITHOUT viene de una clase no_* y no de ausencia de detección


@dataclass(frozen=True)
class ItemStatus:
    state: State
    explicit: bool = False                 # el WITHOUT incluye evidencia explícita en la ventana
    without_since: float | None = None     # inicio del WITHOUT continuo (monotonic)
    confidence: float = 0.0                # confianza media de las observaciones WITHOUT en la ventana

    def without_duration(self, now: float) -> float:
        return 0.0 if self.without_since is None else now - self.without_since


@dataclass(frozen=True)
class PersonPPE:
    track_id: int
    helmet: ItemStatus
    vest: ItemStatus


@dataclass
class PPEConfig:
    with_conf: float = 0.40           # confianza mínima para aceptar EPI puesto
    without_conf: float = 0.40        # confianza mínima para aceptar no_* explícito
    min_person_height_px: int = 90    # más pequeño -> no se puede juzgar casco/chaleco
    max_occlusion: float = 0.30       # fracción del bbox tapada por otra persona
    max_aspect: float = 1.2           # w/h mayor -> postura rara (agachado/tumbado): desconocido
    edge_margin_px: int = 3           # bbox pegado al borde superior -> cabeza probablemente cortada
    window_s: float = 3.0             # ventana temporal de decisión
    min_observations: int = 3         # observaciones válidas mínimas en la ventana
    inferred_weight: float = 0.5      # peso de "no se ve el EPI" frente a evidencia directa
    without_ratio: float = 0.70       # proporción ponderada de WITHOUT para declarar WITHOUT
    with_ratio: float = 0.30          # por debajo -> WITH
    track_ttl_s: float = 10.0         # olvidar estado de tracks no vistos


def _inter(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _box(p) -> tuple:
    return (p.x1, p.y1, p.x2, p.y2)


def roi(p: TrackedPerson, item: str) -> tuple:
    """Zona donde debe aparecer el EPI, en coordenadas de imagen."""
    h, w = p.y2 - p.y1, p.x2 - p.x1
    mx = 0.10 * w
    if item == "helmet":   # cabeza: parte alta del bbox (+ margen por encima: el casco puede sobresalir)
        return (p.x1 - mx, p.y1 - 0.10 * h, p.x2 + mx, p.y1 + 0.30 * h)
    return (p.x1 - mx, p.y1 + 0.15 * h, p.x2 + mx, p.y1 + 0.70 * h)   # torso


def _center_in(d: Detection, r: tuple) -> bool:
    cx, cy = (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2
    return r[0] <= cx <= r[2] and r[1] <= cy <= r[3]


class PPEAssociator:
    def __init__(self, cfg: PPEConfig):
        self.cfg = cfg

    def _assign(self, people: list[TrackedPerson], dets: list[Detection], item: str) -> dict[int, list[Detection]]:
        """Cada detección de EPI se asigna a UNA sola persona (la de ROI más cercano)."""
        out: dict[int, list[Detection]] = {p.track_id: [] for p in people}
        for d in dets:
            if d.label not in (WITH_LABEL[item], WITHOUT_LABEL[item]):
                continue
            cx, cy = (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2
            best, best_dist = None, float("inf")
            for p in people:
                r = roi(p, item)
                if not _center_in(d, r):
                    continue
                dist = (cx - (r[0] + r[2]) / 2) ** 2 + (cy - (r[1] + r[3]) / 2) ** 2
                if dist < best_dist:
                    best, best_dist = p, dist
            if best is not None:
                out[best.track_id].append(d)
        return out

    def _visible(self, p: TrackedPerson, others: list[TrackedPerson], item: str) -> bool:
        c = self.cfg
        h, w = p.y2 - p.y1, p.x2 - p.x1
        if h < c.min_person_height_px or w / max(h, 1) > c.max_aspect:
            return False
        if item == "helmet" and p.y1 <= c.edge_margin_px:
            return False
        area = _area(_box(p))
        covered = sum(_inter(_box(p), _box(o)) for o in others if o.track_id != p.track_id)
        return area > 0 and covered / area <= c.max_occlusion

    def observe(self, people: list[TrackedPerson], ppe: list[Detection]) -> dict[int, dict[str, Observation]]:
        c = self.cfg
        result: dict[int, dict[str, Observation]] = {p.track_id: {} for p in people}
        for item in ITEMS:
            assigned = self._assign(people, ppe, item)
            for p in people:
                dets = assigned[p.track_id]
                with_d = [d for d in dets if d.label == WITH_LABEL[item] and d.confidence >= c.with_conf]
                wout_d = [d for d in dets if d.label == WITHOUT_LABEL[item] and d.confidence >= c.without_conf]
                best_with = max((d.confidence for d in with_d), default=0.0)
                best_wout = max((d.confidence for d in wout_d), default=0.0)

                if best_with and best_with >= best_wout:
                    obs = Observation(State.WITH, best_with)
                elif best_wout:
                    obs = Observation(State.WITHOUT, best_wout, explicit=True)
                elif self._visible(p, people, item):
                    # Zona visible y sin EPI: evidencia débil (puede ser un fallo del detector)
                    obs = Observation(State.WITHOUT, p.confidence, explicit=False)
                else:
                    obs = Observation(State.UNKNOWN)
                result[p.track_id][item] = obs
        return result


class _ItemHistory:
    def __init__(self):
        self.obs: deque[tuple[float, Observation]] = deque()
        self.state = State.UNKNOWN
        self.without_since: float | None = None

    def update(self, now: float, o: Observation, cfg: PPEConfig) -> ItemStatus:
        self.obs.append((now, o))
        while self.obs and now - self.obs[0][0] > cfg.window_s:
            self.obs.popleft()

        valid = [x for _, x in self.obs if x.state is not State.UNKNOWN]
        state = State.UNKNOWN
        explicit, conf = False, 0.0
        if len(valid) >= cfg.min_observations:
            w_with = sum(1.0 for x in valid if x.state is State.WITH)
            w_without = sum(1.0 if x.explicit else cfg.inferred_weight for x in valid if x.state is State.WITHOUT)
            ratio = w_without / (w_with + w_without) if (w_with + w_without) else 0.5
            if ratio >= cfg.without_ratio:
                state = State.WITHOUT
                wo = [x for x in valid if x.state is State.WITHOUT]
                explicit = any(x.explicit for x in wo)
                conf = sum(x.confidence for x in wo) / len(wo)
            elif ratio <= cfg.with_ratio:
                state = State.WITH

        if state is State.WITHOUT:
            if self.without_since is None:
                self.without_since = now
        else:
            self.without_since = None
        self.state = state
        return ItemStatus(state, explicit, self.without_since, conf)


class PPEMonitor:
    """Estado PPE estable por persona trackeada."""

    def __init__(self, cfg: PPEConfig | None = None):
        self.cfg = cfg or PPEConfig()
        self.associator = PPEAssociator(self.cfg)
        self._hist: dict[int, dict[str, _ItemHistory]] = {}
        self._last_seen: dict[int, float] = {}

    def reset(self) -> None:
        self._hist.clear()
        self._last_seen.clear()

    def update(self, people: list[TrackedPerson], ppe: list[Detection], now: float) -> dict[int, PersonPPE]:
        observations = self.associator.observe(people, ppe)
        out: dict[int, PersonPPE] = {}
        for p in people:
            hist = self._hist.setdefault(p.track_id, {i: _ItemHistory() for i in ITEMS})
            self._last_seen[p.track_id] = now
            st = {i: hist[i].update(now, observations[p.track_id][i], self.cfg) for i in ITEMS}
            out[p.track_id] = PersonPPE(p.track_id, st["helmet"], st["vest"])
        for tid in [t for t, ts in self._last_seen.items() if now - ts > self.cfg.track_ttl_s]:
            self._hist.pop(tid, None)
            self._last_seen.pop(tid, None)
        return out
