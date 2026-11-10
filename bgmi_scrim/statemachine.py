"""Temporal confirmation.

A field's value only changes after the same reading has been seen on N
consecutive samples. The transition is emitted -- never the raw frame
reading -- which is what removes OCR flicker from the event log. A reading
below the confidence threshold is treated as its own value, ``UNKNOWN``; when
that is what gets confirmed, the emitted record carries ``state="unknown"``
and no value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from .events import Event

UNKNOWN = "__unknown__"


@dataclass
class Reading:
    t: float
    frame: int
    value: Any
    confidence: float


@dataclass
class ConfirmedField:
    """Debounce one scalar field and emit confirmed transitions."""

    name: str
    confirm_n: int = 3
    threshold: float = 0.60
    monotonic: Optional[str] = None   # None | "down"
    reset_at: Optional[int] = None    # a jump back up to >= this means a new match
    event_type: Optional[str] = None
    emit_unknown: bool = True

    current: Any = None
    current_conf: float = 0.0
    _run: List[Reading] = field(default_factory=list)
    history: List[Tuple[float, Any]] = field(default_factory=list)
    rejected: int = 0

    def observe(self, t: float, frame: int, value: Any, confidence: float) -> Optional[Event]:
        key = value if (value is not None and confidence >= self.threshold) else UNKNOWN
        if self._run and self._run[-1].value == key:
            self._run.append(Reading(t, frame, key, confidence))
        else:
            self._run = [Reading(t, frame, key, confidence)]
        if len(self._run) < self.confirm_n:
            return None
        if key == self.current:
            return None
        if key != UNKNOWN and not self._accept(key):
            self.rejected += 1
            return None
        head = self._run[0]
        confs = [r.confidence for r in self._run]
        conf = sum(confs) / len(confs)
        self.current = key
        self.current_conf = conf
        if key == UNKNOWN:
            if not self.emit_unknown:
                return None
            self.history.append((head.t, None))
            return Event(t=head.t, type=self.event_type or self.name, confidence=conf,
                         state="unknown", value=None, frame=head.frame,
                         source=f"confirmed:{self.confirm_n}")
        self.history.append((head.t, key))
        return Event(t=head.t, type=self.event_type or self.name, confidence=conf,
                     state="ok", value=key, frame=head.frame,
                     source=f"confirmed:{self.confirm_n}")

    def _accept(self, value: Any) -> bool:
        """Sanity gate. Lobby counters never go up inside a match."""
        if self.monotonic != "down" or self.current in (None, UNKNOWN):
            return True
        try:
            if value <= self.current:
                return True
        except TypeError:
            return True
        if self.reset_at is not None and value >= self.reset_at:
            return True  # new match in the same recording
        return False

    def value_at(self, t: float) -> Any:
        out = None
        for ts, v in self.history:
            if ts <= t:
                out = v
            else:
                break
        return out


class FieldSet:
    """A bag of ConfirmedFields observed together, in timestamp order."""

    def __init__(self) -> None:
        self.fields: dict[str, ConfirmedField] = {}

    def add(self, f: ConfirmedField) -> ConfirmedField:
        self.fields[f.name] = f
        return f

    def observe(self, name: str, t: float, frame: int, value: Any,
                confidence: float) -> Optional[Event]:
        f = self.fields.get(name)
        if f is None:
            return None
        return f.observe(t, frame, value, confidence)

    def __getitem__(self, name: str) -> ConfirmedField:
        return self.fields[name]
