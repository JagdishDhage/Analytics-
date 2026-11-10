"""Event records and the JSONL writer.

Every record carries a timestamp (seconds from the start of the video), the
frame index it was confirmed on, and a confidence in [0, 1]. A record whose
confidence is below the profile threshold is emitted with ``state="unknown"``
and a null value -- never a guess.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, IO, Iterable, List, Optional

# Event type constants
MATCH_START = "match_start"
MATCH_END = "match_end"
MAP_NAME = "map_name"
TEAMS_ALIVE = "teams_alive"
PLAYERS_ALIVE = "players_alive"
TEAM_ALIVE_COUNT = "team_alive_count"
KNOCK = "knock"
KILL = "kill"
ZONE_PHASE = "zone_phase"
ZONE_STATE = "zone_state"
PLACEMENT = "placement"
WINNER = "winner"

COUNTER_TYPES = (TEAMS_ALIVE, PLAYERS_ALIVE, ZONE_PHASE)


@dataclass
class Event:
    t: float
    type: str
    confidence: float
    state: str = "ok"           # "ok" | "unknown"
    value: Any = None           # counter value / map name / placement / winner
    frame: Optional[int] = None
    source: str = ""            # which detector produced it
    team: Optional[str] = None  # for team_alive_count
    victim_team: Optional[str] = None
    victim_slot: Optional[int] = None
    killer_team: Optional[str] = None
    killer_slot: Optional[int] = None
    zone_phase: Optional[int] = None
    match_index: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        d = {k: v for k, v in asdict(self).items() if v not in (None, "", {})}
        d["t"] = round(float(self.t), 3)
        d["type"] = self.type
        d["confidence"] = round(float(self.confidence), 4)
        d["state"] = self.state
        if self.value is not None:
            d["value"] = self.value
        return json.dumps(d, sort_keys=True)


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh: Optional[IO[str]] = None
        self.count = 0

    def __enter__(self) -> "JsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w")
        return self

    def write(self, ev: Event) -> None:
        assert self._fh is not None
        self._fh.write(ev.to_json() + "\n")
        self.count += 1

    def write_all(self, evs: Iterable[Event]) -> None:
        for ev in evs:
            self.write(ev)

    def __exit__(self, *exc: Any) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
