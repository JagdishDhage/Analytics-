"""Fusion of panel-derived and feed-derived knock/kill events.

The team-list panel is the reliable witness that a player went down: it is a
colour read, not OCR, and it is confirmed over N samples. The kill feed is the
only witness to *who* did it, and it is OCR, so it is used for attribution
rather than for existence. A feed event with no panel counterpart is emitted
only where the panel has no opinion about that victim; otherwise it is
suppressed and counted, which is what keeps phantom kills out of the log.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .events import Event, KILL, KNOCK
from .killfeed import FeedEvent


@dataclass
class PanelEvent:
    t: float
    frame: int
    kind: str
    victim_team: Optional[str]
    victim_slot: Optional[int]
    confidence: float
    zone_phase: Optional[int] = None
    match_index: Optional[int] = None


@dataclass
class Fuser:
    fuse_window_s: float = 6.0
    panel_is_authority: bool = True
    # (victim_team, victim_slot, t) -> True if the panel saw that slot change
    # state near t, or has no confirmed opinion about it at all.
    corroborates: Callable[[Optional[str], Optional[int], float], bool] = lambda a, b, c: True
    _panel: List[PanelEvent] = field(default_factory=list)
    _feed: List[FeedEvent] = field(default_factory=list)
    suppressed_feed: List[FeedEvent] = field(default_factory=list)
    attributed: int = 0

    def add_panel(self, ev: PanelEvent) -> None:
        self._panel.append(ev)

    def add_feed(self, ev: FeedEvent) -> None:
        self._feed.append(ev)

    def _match_feed(self, pe: PanelEvent) -> Optional[FeedEvent]:
        best, best_dt = None, None
        for fe in self._feed:
            if fe.kind != pe.kind:
                continue
            if pe.victim_team is not None and fe.victim_team != pe.victim_team:
                continue
            if (pe.victim_slot is not None and fe.victim_slot is not None
                    and fe.victim_slot != pe.victim_slot):
                continue
            dt = abs(fe.t - pe.t)
            if dt > self.fuse_window_s:
                continue
            if best_dt is None or dt < best_dt:
                best, best_dt = fe, dt
        return best

    def flush(self, now: Optional[float] = None) -> List[Event]:
        """Emit every event that can no longer gain attribution."""
        out: List[Event] = []
        ready = [p for p in self._panel
                 if now is None or (now - p.t) > self.fuse_window_s]
        for pe in ready:
            fe = self._match_feed(pe)
            killer_team = killer_slot = None
            conf = pe.confidence
            source = "panel"
            if fe is not None:
                self._feed.remove(fe)
                killer_team, killer_slot = fe.killer_team, fe.killer_slot
                source = "panel+feed"
                if killer_team is not None:
                    self.attributed += 1
                conf = min(1.0, 0.5 * pe.confidence + 0.5 * fe.confidence + 0.1)
            out.append(Event(
                t=pe.t, type=pe.kind, confidence=conf,
                state="ok" if conf > 0 else "unknown", frame=pe.frame, source=source,
                victim_team=pe.victim_team, victim_slot=pe.victim_slot,
                killer_team=killer_team, killer_slot=killer_slot,
                zone_phase=pe.zone_phase, match_index=pe.match_index))
            self._panel.remove(pe)

        stale = [f for f in self._feed
                 if now is None or (now - f.t) > self.fuse_window_s]
        for fe in stale:
            self._feed.remove(fe)
            if self.panel_is_authority and not self.corroborates(
                    fe.victim_team, fe.victim_slot, fe.t):
                self.suppressed_feed.append(fe)   # panel disagreed: not emitted
                continue
            out.append(Event(
                t=fe.t, type=fe.kind, confidence=fe.confidence * 0.9, state="ok",
                source="feed", victim_team=fe.victim_team, victim_slot=fe.victim_slot,
                killer_team=fe.killer_team, killer_slot=fe.killer_slot))
        return out
