"""Fights and eliminations read from positions alone.

A map stream with labelled dots and no kill feed never says who killed whom.
What it does say is where every team was, and when a dot stopped existing. So
a fight here is not an event anyone announced: it is two teams close together,
and players lost while they were close.

That is weaker than a kill feed and the vocabulary keeps it honest. A player
"lost" is a dot that stopped appearing, which is usually death and sometimes
the tracker losing a dot under map furniture. Nothing here claims an attacker.
What it can say, because it has both teams' paths, is who closed the distance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULTS: Dict[str, float] = {
    #: two teams are in contact when any pair of their players is this close
    "contact_m": 120.0,
    #: contact ends when they are further than contact_m + this for gap_s
    "release_m": 60.0,
    "gap_s": 20.0,
    #: a contact shorter than this is two teams passing, not a fight
    "min_duration_s": 6.0,
    #: losses this long after a contact ends still belong to it
    "tail_s": 25.0,
    #: a player whose dot stops this long before the end of the stream is
    #: treated as lost; one that stops later may simply have outlived the clip
    "loss_grace_s": 25.0,
    #: a dot missing for less than this and then seen again never left
    "reacquire_s": 12.0,
    #: window before contact used to decide who closed the distance
    "approach_s": 30.0,
    #: 90th-percentile speed above this reads as a vehicle, as elsewhere
    "vehicle_mps": 11.0,
    #: grid the paths are resampled onto before pairing teams up
    "step_s": 1.0,
    #: centroid pre-filter: pairs further apart than this skip exact pairing
    "coarse_m": 600.0,
}


@dataclass
class Path:
    """One player's positions, in time order."""

    team: str
    player: Optional[str]
    t: List[float] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    identity: List[float] = field(default_factory=list)

    @property
    def t0(self) -> float:
        return self.t[0]

    @property
    def t1(self) -> float:
        return self.t[-1]

    def at(self, t: float, hold: float) -> Optional[Tuple[float, float]]:
        """Position at t, interpolated, or None where the dot was not seen.

        `hold` bounds how far a sample is allowed to speak for: a dot seen at
        10 s says nothing about 400 s.
        """
        if not self.t or t < self.t[0] - hold or t > self.t[-1] + hold:
            return None
        lo, hi = 0, len(self.t) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.t[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        i = lo
        if self.t[i] == t or i == 0:
            if abs(self.t[i] - t) > hold:
                return None
            return self.x[i], self.y[i]
        a, b = i - 1, i
        span = self.t[b] - self.t[a]
        if span <= 0:
            return self.x[b], self.y[b]
        if span > hold * 2:
            # a gap this wide is not a path, it is two paths
            near = a if (t - self.t[a]) < (self.t[b] - t) else b
            if abs(self.t[near] - t) > hold:
                return None
            return self.x[near], self.y[near]
        f = (t - self.t[a]) / span
        return (self.x[a] + f * (self.x[b] - self.x[a]),
                self.y[a] + f * (self.y[b] - self.y[a]))

    def speed_p90(self, t0: float, t1: float) -> Optional[float]:
        sp: List[float] = []
        for i in range(1, len(self.t)):
            if self.t[i] < t0 or self.t[i] > t1:
                continue
            dt = self.t[i] - self.t[i - 1]
            if dt <= 0 or dt > 5.0:
                continue
            d = math.dist((self.x[i], self.y[i]), (self.x[i - 1], self.y[i - 1]))
            sp.append(d / dt)
        if not sp:
            return None
        sp.sort()
        return sp[min(len(sp) - 1, int(0.9 * len(sp)))]


@dataclass
class Loss:
    team: str
    player: Optional[str]
    t: float
    confidence: float
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return {"type": "player_lost", "t": round(self.t, 2), "team": self.team,
                "player": self.player, "confidence": round(self.confidence, 3),
                "reason": self.reason}


@dataclass
class Contact:
    t0: float
    t1: float
    team_a: str
    team_b: str
    min_dist_m: float
    losses_a: int
    losses_b: int
    #: metres each side closed in the window before contact; None where the
    #: team was not tracked for long enough to say
    closed_a: Optional[float]
    closed_b: Optional[float]
    mode_a: Optional[str]
    mode_b: Optional[str]
    players_a: int
    players_b: int

    @property
    def initiator(self) -> Optional[str]:
        """Whichever side closed more ground. Not who shot first."""
        if self.closed_a is None or self.closed_b is None:
            return None
        if abs(self.closed_a - self.closed_b) < 50.0:
            return None       # both moved in, or neither did
        return self.team_a if self.closed_a > self.closed_b else self.team_b

    def outcome_for(self, team: str) -> Optional[str]:
        ours, theirs = ((self.losses_a, self.losses_b) if team == self.team_a
                        else (self.losses_b, self.losses_a))
        if ours == 0 and theirs == 0:
            return "no losses"
        if ours == theirs:
            return "even"
        return "won" if theirs > ours else "lost"

    def as_dict(self) -> Dict[str, Any]:
        return {"type": "contact", "t": round(self.t0, 2), "t_end": round(self.t1, 2),
                "duration_s": round(self.t1 - self.t0, 1),
                "team": self.team_a, "opponent": self.team_b,
                "min_distance_m": round(self.min_dist_m, 1),
                "losses_team": self.losses_a, "losses_opponent": self.losses_b,
                "closed_team_m": None if self.closed_a is None else round(self.closed_a),
                "closed_opponent_m": None if self.closed_b is None else round(self.closed_b),
                "mode_team": self.mode_a, "mode_opponent": self.mode_b,
                "players_team": self.players_a, "players_opponent": self.players_b,
                "initiator": self.initiator}


def paths_from_samples(samples: Iterable[Dict[str, Any]],
                       min_identity: float = 0.0) -> List[Path]:
    """Group position samples into one path per named player.

    Grouping by (team, player) rather than by track id on purpose: the tracker
    starts a new track whenever it loses a dot, and three tracks of one player
    are still one player.
    """
    by_key: Dict[Tuple[str, Optional[str]], Path] = {}
    rows = []
    for s in samples:
        if s.get("type") != "sample":
            continue
        team = s.get("team")
        if not team:
            continue
        if float(s.get("identity_confidence") or 0.0) < min_identity:
            continue
        rows.append(s)
    rows.sort(key=lambda r: float(r["t"]))
    for s in rows:
        key = (str(s["team"]), s.get("player"))
        p = by_key.get(key)
        if p is None:
            p = by_key[key] = Path(str(s["team"]), s.get("player"))
        t = float(s["t"])
        if p.t and t == p.t[-1]:
            continue
        p.t.append(t)
        p.x.append(float(s["x_m"]))
        p.y.append(float(s["y_m"]))
        p.identity.append(float(s.get("identity_confidence") or 0.0))
    return [p for p in by_key.values() if len(p.t) >= 2]


def losses(paths: Sequence[Path], match_end: float,
           cfg: Optional[Dict[str, float]] = None) -> List[Loss]:
    """Players whose dot stopped appearing well before the stream ended."""
    c = {**DEFAULTS, **(cfg or {})}
    out: List[Loss] = []
    for p in paths:
        if p.player is None:
            continue
        if p.t1 >= match_end - c["loss_grace_s"]:
            continue    # still on the map when the stream ran out
        ident = sum(p.identity) / len(p.identity) if p.identity else 0.0
        seen = len(p.t)
        # a long path that ends cleanly is a better death signal than a
        # two-sample path that was probably never really tracked
        conf = min(0.9, 0.35 + 0.25 * min(1.0, seen / 40.0) + 0.3 * ident)
        out.append(Loss(p.team, p.player, p.t1, conf, "dot stopped appearing"))
    out.sort(key=lambda l: l.t)
    return out


def team_eliminations(paths: Sequence[Path], match_end: float,
                      cfg: Optional[Dict[str, float]] = None
                      ) -> Dict[str, Optional[float]]:
    """When each team's last dot went, or None where it outlived the stream."""
    c = {**DEFAULTS, **(cfg or {})}
    last: Dict[str, float] = {}
    for p in paths:
        last[p.team] = max(last.get(p.team, 0.0), p.t1)
    return {team: (None if t >= match_end - c["loss_grace_s"] else t)
            for team, t in last.items()}


def placements(elims: Dict[str, Optional[float]]) -> Dict[str, int]:
    """Rank teams by when they died: last out is 1st.

    Teams still alive when the stream ends cannot be separated, so they share
    the best rank rather than being ordered on nothing.
    """
    alive = [t for t, e in elims.items() if e is None]
    dead = sorted(((e, t) for t, e in elims.items() if e is not None), reverse=True)
    out: Dict[str, int] = {}
    rank = 1
    for t in sorted(alive):
        out[t] = 1
    if alive:
        rank = 1 + len(alive)
    for _, t in dead:
        out[t] = rank
        rank += 1
    return out


def _centroid(ps: Sequence[Path], t: float, hold: float) -> Optional[Tuple[float, float, int]]:
    xs, ys = [], []
    for p in ps:
        pos = p.at(t, hold)
        if pos is not None:
            xs.append(pos[0])
            ys.append(pos[1])
    if not xs:
        return None
    return sum(xs) / len(xs), sum(ys) / len(ys), len(xs)


def _min_pair(a: Sequence[Path], b: Sequence[Path], t: float,
              hold: float) -> Optional[Tuple[float, int, int]]:
    pa = [p.at(t, hold) for p in a]
    pb = [p.at(t, hold) for p in b]
    pa = [q for q in pa if q is not None]
    pb = [q for q in pb if q is not None]
    if not pa or not pb:
        return None
    best = min(math.dist(q, r) for q in pa for r in pb)
    return best, len(pa), len(pb)


def contacts(paths: Sequence[Path], match_end: float,
             cfg: Optional[Dict[str, float]] = None) -> List[Contact]:
    """Windows where two teams were close, with what each side lost."""
    c = {**DEFAULTS, **(cfg or {})}
    by_team: Dict[str, List[Path]] = {}
    for p in paths:
        by_team.setdefault(p.team, []).append(p)
    teams = sorted(by_team)
    if len(teams) < 2:
        return []
    hold = c["step_s"] * 3.0
    grid = [t for t in _frange(0.0, match_end, c["step_s"])]
    all_losses = losses(paths, match_end, c)

    # centroids once per team per step, so the pair loop stays cheap
    cents: Dict[str, Dict[float, Tuple[float, float, int]]] = {t: {} for t in teams}
    for team in teams:
        for t in grid:
            ct = _centroid(by_team[team], t, hold)
            if ct is not None:
                cents[team][t] = ct

    out: List[Contact] = []
    for i, ta in enumerate(teams):
        for tb in teams[i + 1:]:
            open_t: Optional[float] = None
            last_close: Optional[float] = None
            mind = float("inf")
            pa_max = pb_max = 0
            for t in grid:
                ca, cb = cents[ta].get(t), cents[tb].get(t)
                near = False
                if ca is not None and cb is not None and math.dist(ca[:2], cb[:2]) <= c["coarse_m"]:
                    mp = _min_pair(by_team[ta], by_team[tb], t, hold)
                    if mp is not None:
                        d, na, nb = mp
                        if d <= c["contact_m"]:
                            near = True
                            if open_t is None:
                                open_t = t
                                mind = d
                                pa_max = pb_max = 0
                            mind = min(mind, d)
                            pa_max = max(pa_max, na)
                            pb_max = max(pb_max, nb)
                            last_close = t
                        elif (open_t is not None and d <= c["contact_m"] + c["release_m"]):
                            # drifting apart but not yet separated
                            near = True
                if not near and open_t is not None and last_close is not None:
                    if t - last_close >= c["gap_s"]:
                        ct = _close(ta, tb, open_t, last_close, mind, pa_max, pb_max,
                                    by_team, all_losses, c)
                        if ct is not None:
                            out.append(ct)
                        open_t = None
                        last_close = None
                        mind = float("inf")
            if open_t is not None and last_close is not None:
                ct = _close(ta, tb, open_t, last_close, mind, pa_max, pb_max,
                            by_team, all_losses, c)
                if ct is not None:
                    out.append(ct)
    out.sort(key=lambda k: (k.t0, k.team_a, k.team_b))
    return out


def _close(ta: str, tb: str, t0: float, t1: float, mind: float,
           na: int, nb: int, by_team: Dict[str, List[Path]],
           all_losses: Sequence[Loss], c: Dict[str, float]) -> Optional[Contact]:
    if t1 - t0 < c["min_duration_s"]:
        return None
    lo, hi = t0 - 2.0, t1 + c["tail_s"]
    la = sum(1 for l in all_losses if l.team == ta and lo <= l.t <= hi)
    lb = sum(1 for l in all_losses if l.team == tb and lo <= l.t <= hi)
    closed_a = _closed(by_team[ta], by_team[tb], t0, c)
    closed_b = _closed(by_team[tb], by_team[ta], t0, c)
    return Contact(t0, t1, ta, tb, mind, la, lb, closed_a, closed_b,
                   _mode(by_team[ta], t0, c), _mode(by_team[tb], t0, c), na, nb)


def _closed(mine: Sequence[Path], theirs: Sequence[Path], t0: float,
            c: Dict[str, float]) -> Optional[float]:
    """Metres my side took off the gap in the window before contact."""
    hold = c["step_s"] * 3.0
    t_start = t0 - c["approach_s"]
    a0 = _centroid(mine, t_start, hold)
    a1 = _centroid(mine, t0, hold)
    b0 = _centroid(theirs, t_start, hold)
    b1 = _centroid(theirs, t0, hold)
    if not all((a0, a1, b0, b1)):
        return None
    # distance from my start to their start, against my end to their start:
    # how much of the closing I did, holding their position fixed
    before = math.dist(a0[:2], b0[:2])
    after = math.dist(a1[:2], b0[:2])
    return before - after


def _mode(ps: Sequence[Path], t0: float, c: Dict[str, float]) -> Optional[str]:
    sp = [p.speed_p90(t0 - c["approach_s"], t0) for p in ps]
    sp = [s for s in sp if s is not None]
    if not sp:
        return None
    return "vehicle" if max(sp) > c["vehicle_mps"] else "foot"


def _frange(a: float, b: float, step: float) -> Iterable[float]:
    n = int(math.floor((b - a) / step)) + 1
    for i in range(max(0, n)):
        yield a + i * step
