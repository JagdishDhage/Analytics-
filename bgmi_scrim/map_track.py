"""Multi-target tracking of map dots, and who each track is.

Reading every name on every frame is not affordable: 100 dots at 2 fps is
200 OCR calls a second of footage. Identity is therefore resolved sparsely --
labels are read on a schedule and for tracks that do not have one yet -- and
carried between reads by tracking, exactly as the team-list panel's row tags
are in phase 1. A track with no confirmed label stays anonymous rather than
being guessed into a name.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class Detection:
    x: float                 # world metres
    y: float
    cluster: int = 1
    confidence: float = 0.9
    team: Optional[str] = None
    player: Optional[str] = None
    label_conf: float = 0.0
    team_conf: float = 0.0
    colour: Tuple[int, int, int] = (0, 0, 0)


@dataclass
class Track:
    id: int
    x: float
    y: float
    t: float
    first_t: float
    last_seen_t: float
    samples: int = 1
    missed: int = 0
    cluster: int = 1
    speed_mps: float = 0.0
    confidence: float = 0.9
    colour: Tuple[int, int, int] = (0, 0, 0)
    votes: Dict[Tuple[str, str], float] = field(default_factory=dict)
    team_votes: Dict[str, float] = field(default_factory=dict)

    def identity(self) -> Tuple[Optional[str], Optional[str], float]:
        """Team first, player only when the name itself was read.

        The team survives a crowded label and a colour read; the player does
        not, and reporting a player the log cannot support would be a guess.
        """
        team = player = None
        conf = 0.0
        if self.team_votes:
            team, w = max(self.team_votes.items(), key=lambda kv: kv[1])
            total = sum(self.team_votes.values())
            conf = (w / total) if total else 0.0
        if self.votes:
            (pteam, pplayer), w = max(self.votes.items(), key=lambda kv: kv[1])
            total = sum(self.votes.values())
            pconf = (w / total) if total else 0.0
            if team is None or pteam == team:
                team, player = pteam, pplayer or None
                conf = max(conf, pconf)
        return team, player, conf

    def vote(self, team: str, player: Optional[str], weight: float,
             team_weight: float = 0.0) -> None:
        if player:
            self.votes[(team, player)] = self.votes.get((team, player), 0.0) + max(weight, 0.01)
        if team:
            self.team_votes[team] = self.team_votes.get(team, 0.0) + max(team_weight or weight, 0.01)


class Tracker:
    def __init__(self, max_speed_mps: float = 140.0, max_miss_s: float = 8.0,
                 gate_floor_m: float = 60.0):
        self.max_speed = max_speed_mps
        self.max_miss_s = max_miss_s
        self.gate_floor = gate_floor_m
        self.tracks: List[Track] = []
        self._all: Dict[int, Track] = {}
        self._next_id = 1
        self.id_switch_guard = 0

    def all_tracks(self) -> List[Track]:
        """Every track the run produced, including ones that have ended."""
        return list(self._all.values())

    # ---------------- stepping ----------------
    def update(self, t: float, dets: Sequence[Detection]) -> List[Track]:
        dt = max(1e-3, t - max((tr.t for tr in self.tracks), default=t))
        gate = max(self.gate_floor, self.max_speed * dt)
        pairs: List[Tuple[float, int, int]] = []
        for di, d in enumerate(dets):
            for ti, tr in enumerate(self.tracks):
                dist = math.dist((d.x, d.y), (tr.x, tr.y))
                if dist > gate:
                    continue
                cost = dist
                if d.team is not None:
                    team, player, _ = tr.identity()
                    if team is not None:
                        cost = dist * (0.4 if (team, player) == (d.team, d.player) else 2.5)
                pairs.append((cost, di, ti))
        pairs.sort()
        used_d: set = set()
        used_t: set = set()
        for cost, di, ti in pairs:
            if di in used_d or ti in used_t:
                continue
            used_d.add(di)
            used_t.add(ti)
            self._apply(self.tracks[ti], t, dets[di])
        for di, d in enumerate(dets):
            if di in used_d:
                continue
            tr = Track(id=self._next_id, x=d.x, y=d.y, t=t, first_t=t, last_seen_t=t,
                       cluster=d.cluster, confidence=d.confidence, colour=d.colour)
            self._next_id += 1
            if d.team is not None:
                tr.vote(d.team, d.player, d.label_conf, d.team_conf)
            self.tracks.append(tr)
            self._all[tr.id] = tr
        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.missed += 1
        self.tracks = [tr for tr in self.tracks
                       if (t - tr.last_seen_t) <= self.max_miss_s]
        return [tr for ti, tr in enumerate(self.tracks) if tr.last_seen_t == t]

    def _apply(self, tr: Track, t: float, d: Detection) -> None:
        dt = max(1e-3, t - tr.t)
        tr.speed_mps = math.dist((d.x, d.y), (tr.x, tr.y)) / dt
        tr.x, tr.y, tr.t, tr.last_seen_t = d.x, d.y, t, t
        tr.samples += 1
        tr.missed = 0
        tr.cluster = d.cluster
        tr.confidence = d.confidence
        tr.colour = d.colour
        if d.team is not None:
            tr.vote(d.team, d.player, d.label_conf, d.team_conf)

    # ---------------- identity reconciliation ----------------
    def resolve_frame(self, live: Sequence[Track]) -> Dict[int, Tuple[Optional[str], Optional[str], float]]:
        """One identity per track, and no two live tracks claiming one player."""
        claims: List[Tuple[float, int, str, str]] = []
        out: Dict[int, Tuple[Optional[str], Optional[str], float]] = {}
        for tr in live:
            team, player, conf = tr.identity()
            out[tr.id] = (None, None, 0.0)
            if team is not None:
                claims.append((sum(tr.votes.values()) * conf, tr.id, team, player or ""))
        claims.sort(reverse=True)
        taken: set = set()
        for weight, tid, team, player in claims:
            if (team, player) in taken:
                self.id_switch_guard += 1
                continue
            taken.add((team, player))
            tr = next(t for t in live if t.id == tid)
            _, _, conf = tr.identity()
            out[tid] = (team, player, conf)
        return out

    def needs_label(self, tr: Track) -> bool:
        team, _, conf = tr.identity()
        return team is None or conf < 0.6
