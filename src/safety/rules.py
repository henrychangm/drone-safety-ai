"""Reglas de incumplimiento: convierte estado PPE sostenido en incidencias candidatas.

Una detección aislada nunca basta: hace falta que el estado WITHOUT dure el tiempo exigido
(ya filtrado temporalmente por PPEMonitor). Si a la vez falta el otro EPI, va en la MISMA
incidencia ("no_helmet,no_vest") para que el revisor vea una sola tarjeta por persona.
Tras reportar, cada (persona, EPI) queda en cooldown.
"""
from dataclasses import dataclass

from src.safety.ppe import ItemStatus, PersonPPE, State

ITEM_TYPE = {"helmet": "no_helmet", "vest": "no_vest"}


@dataclass
class RulesConfig:
    violation_duration_s: float = 5.0        # duración mínima con evidencia explícita (clase no_*)
    helmet_inferred_duration_s: float = 8.0  # casco: solo se infiere por ausencia
    vest_inferred_duration_s: float = 10.0   # chaleco: el dataset no tiene no_vest, siempre inferido
    incident_cooldown_s: float = 60.0        # no repetir (track, EPI) durante este tiempo
    require_explicit_helmet: bool = False    # True: casco solo con "no_helmet" explícito


@dataclass(frozen=True)
class Violation:
    track_id: int
    type: str            # "no_helmet" | "no_vest" | "no_helmet,no_vest"
    confidence: float
    duration_s: float
    explicit: bool


class RuleEngine:
    def __init__(self, cfg: RulesConfig | None = None):
        self.cfg = cfg or RulesConfig()
        self._last_reported: dict[tuple[int, str], float] = {}

    def reset(self) -> None:
        self._last_reported.clear()

    def _required_duration(self, item: str, st: ItemStatus) -> float | None:
        """Duración exigida, o None si esta evidencia nunca es suficiente."""
        if st.explicit:
            return self.cfg.violation_duration_s
        if item == "helmet":
            return None if self.cfg.require_explicit_helmet else self.cfg.helmet_inferred_duration_s
        return self.cfg.vest_inferred_duration_s

    def _in_cooldown(self, tid: int, item: str, now: float) -> bool:
        last = self._last_reported.get((tid, ITEM_TYPE[item]))
        return last is not None and now - last < self.cfg.incident_cooldown_s

    def evaluate(self, status: dict[int, PersonPPE], now: float) -> list[Violation]:
        out: list[Violation] = []
        for tid, person in status.items():
            items = {"helmet": person.helmet, "vest": person.vest}
            due = []
            for item, st in items.items():
                if st.state is not State.WITHOUT or self._in_cooldown(tid, item, now):
                    continue
                need = self._required_duration(item, st)
                if need is not None and st.without_duration(now) >= need:
                    due.append(item)
            if not due:
                continue
            # El otro EPI que también falte (estado ya estable) se adjunta a la misma incidencia.
            included = [i for i, st in items.items()
                        if (i in due) or (st.state is State.WITHOUT and not self._in_cooldown(tid, i, now))]
            for i in included:
                self._last_reported[(tid, ITEM_TYPE[i])] = now
            sts = [items[i] for i in included]
            out.append(Violation(
                tid, ",".join(ITEM_TYPE[i] for i in ("helmet", "vest") if i in included),
                sum(s.confidence for s in sts) / len(sts),
                max(s.without_duration(now) for s in sts),
                any(s.explicit for s in sts)))
        self._last_reported = {k: t for k, t in self._last_reported.items()
                               if now - t < self.cfg.incident_cooldown_s}
        return out
