"""Turning position tracks into behaviour a coach can act on.

Everything here is derived from positions and zone circles and nothing else.
Each definition is stated in the report so a number can be argued with:

* landing   -- first time a player's speed stays under ``land_speed_mps``
* looting   -- a stretch under ``loot_speed_mps`` lasting at least ``loot_min_s``
* rotation  -- a stretch over ``move_speed_mps`` covering at least ``rotation_min_m``
* vehicle   -- a rotation whose 90th-percentile speed exceeds ``vehicle_speed_mps``
* zone phase-- the safe circle changing is the announcement; the blue circle
               reaching the safe circle's radius is the close
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULTS = {
    "land_speed_mps": 12.0, "land_hold_s": 2.0, "air_speed_mps": 25.0,
    "drop_window_s": 150.0,
    "max_sample_gap_s": 10.0,
    "loot_speed_mps": 2.5, "loot_min_s": 15.0,
    "move_speed_mps": 3.0, "rotation_min_m": 200.0, "rotation_gap_s": 6.0,
    "rotation_min_displacement_m": 150.0,
    "vehicle_speed_mps": 11.0, "edge_band": 0.25,
}


@dataclass
class Sample:
    t: float
    x: float
    y: float
    confidence: float = 1.0
    cluster: int = 1


@dataclass
class ZoneSample:
    t: float
    ring: str
    cx: float
    cy: float
    r: float
    confidence: float = 1.0


@dataclass
class Phase:
    number: int
    announced_t: float
    cx: float
    cy: float
    r: float
    closed_t: Optional[float] = None


@dataclass
class Segment:
    kind: str                 # "loot" | "rotation"
    t0: float
    t1: float
    x0: float
    y0: float
    x1: float
    y1: float
    distance_m: float
    mean_speed: float
    peak_speed: float
    mode: str = ""            # foot | vehicle (rotations only)
    displacement_m: float = 0.0
    continued: bool = False   # this part carried on from the previous phase

    @property
    def duration(self) -> float:
        return max(0.0, self.t1 - self.t0)


def speeds(samples: Sequence[Sample]) -> List[float]:
    out = [0.0]
    for a, b in zip(samples, samples[1:]):
        dt = max(1e-3, b.t - a.t)
        out.append(math.dist((a.x, a.y), (b.x, b.y)) / dt)
    return out


def smooth(values: Sequence[float], k: int = 3) -> List[float]:
    if k <= 1 or len(values) < k:
        return list(values)
    out = []
    half = k // 2
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def landing(samples: Sequence[Sample], cfg: Dict[str, Any],
            t0: Optional[float] = None) -> Tuple[Optional[Sample], bool]:
    """Where a unit came down, and whether a descent was actually seen.

    A dot sitting still in the aircraft is slow, so "first slow moment" lands
    everyone in the plane. The descent is the giveaway: nothing on the ground
    crosses the map at air speed, so landing is the first slow moment *after*
    the last fast one. When the stream starts after the drop there is no
    descent to find, and that is reported rather than papered over.
    """
    if not samples:
        return None, False
    sp = smooth(speeds(samples))
    hold = float(cfg.get("land_hold_s", DEFAULTS["land_hold_s"]))
    limit = float(cfg.get("land_speed_mps", DEFAULTS["land_speed_mps"]))
    air = float(cfg.get("air_speed_mps", DEFAULTS["air_speed_mps"]))
    # The drop happens once, early. Only a descent inside the drop window
    # counts, so a tracking jump in the middle of a match cannot be mistaken
    # for one.
    start = samples[0].t if t0 is None else t0
    window = float(cfg.get("drop_window_s", DEFAULTS["drop_window_s"]))
    first_air = next((i for i, v in enumerate(sp)
                      if v >= air and samples[i].t <= start + window), None)
    if first_air is None:
        return samples[0], False
    # Landing is where the descent ends, not where the squad finally stops:
    # after touching down a squad keeps moving, and waiting for it to be still
    # puts the drop minutes late and in the wrong place.
    for i in range(first_air + 1, len(samples)):
        if sp[i] < air:
            return samples[i], True
    return samples[-1], True


def segments(samples: Sequence[Sample], cfg: Dict[str, Any]) -> List[Segment]:
    """Split a track into looting (slow) and rotation (moving) stretches."""
    if len(samples) < 2:
        return []
    c = {**DEFAULTS, **(cfg or {})}
    sp = smooth(speeds(samples))
    out: List[Segment] = []
    i = 0
    n = len(samples)
    max_gap = float(c.get("max_sample_gap_s", DEFAULTS["max_sample_gap_s"]))
    while i < n:
        moving = sp[i] >= c["move_speed_mps"]
        j = i + 1
        gap = 0.0
        while j < n:
            # a hole in the tracking is not movement: never span one
            if samples[j].t - samples[j - 1].t > max_gap:
                break
            is_move = sp[j] >= c["move_speed_mps"]
            if is_move == moving:
                gap = 0.0
            else:
                gap += samples[j].t - samples[j - 1].t
                if gap > c["rotation_gap_s"]:
                    break
            j += 1
        end = max(i, j - 1)          # the last sample that belongs to this stretch
        a, b = samples[i], samples[end]
        seg_sp = sp[i:end + 1]
        dist = sum(math.dist((samples[k].x, samples[k].y), (samples[k + 1].x, samples[k + 1].y))
                   for k in range(i, end))
        dur = max(1e-3, b.t - a.t)
        seg = Segment("rotation" if moving else "loot", a.t, b.t, a.x, a.y, b.x, b.y,
                      dist, dist / dur, max(seg_sp) if seg_sp else 0.0,
                      displacement_m=math.dist((a.x, a.y), (b.x, b.y)))
        if moving:
            # Path length alone calls jitter and a fight in one compound a
            # rotation. A rotation also has to end up somewhere else.
            if (dist >= c["rotation_min_m"]
                    and seg.displacement_m >= c["rotation_min_displacement_m"]):
                p90 = sorted(seg_sp)[int(0.9 * (len(seg_sp) - 1))] if seg_sp else 0.0
                seg.mode = "vehicle" if p90 >= c["vehicle_speed_mps"] else "foot"
                out.append(seg)
        elif seg.duration >= c["loot_min_s"]:
            out.append(seg)
        i = max(j, i + 1)
    return out


def _median_filter(zs: Sequence[ZoneSample], k: int = 3) -> List[ZoneSample]:
    """Median of a sliding window, so one noisy fit cannot declare a phase."""
    if k <= 1 or len(zs) < k:
        return list(zs)
    out: List[ZoneSample] = []
    half = k // 2
    for i, z in enumerate(zs):
        lo, hi = max(0, i - half), min(len(zs), i + half + 1)
        win = zs[lo:hi]
        xs = sorted(w.cx for w in win)
        ys = sorted(w.cy for w in win)
        rs = sorted(w.r for w in win)
        m = len(win) // 2
        out.append(ZoneSample(z.t, z.ring, xs[m], ys[m], rs[m], z.confidence))
    return out


def split_at(seg: Segment, samples: Sequence[Sample], cuts: Sequence[float],
             cfg: Dict[str, Any]) -> List[Segment]:
    """Cut a movement at the given times, recomputing each part from the track.

    A squad that keeps driving through two circle announcements has not made
    one rotation, it has made the rotation for each of those circles, and a
    coach reads them per phase. Splitting at the announcements is what lets
    the delay and the margin be measured against the right circle.
    """
    c = {**DEFAULTS, **(cfg or {})}
    inner = sorted(x for x in cuts if seg.t0 + 1e-6 < x < seg.t1 - 1e-6)
    if not inner:
        return [seg]
    bounds = [seg.t0] + inner + [seg.t1]
    parts: List[Segment] = []
    for a, b in zip(bounds, bounds[1:]):
        sub = [s for s in samples if a - 1e-9 <= s.t <= b + 1e-9]
        if len(sub) < 2:
            continue
        dist = sum(math.dist((p.x, p.y), (q.x, q.y)) for p, q in zip(sub, sub[1:]))
        dur = max(1e-3, sub[-1].t - sub[0].t)
        sp = smooth(speeds(sub))
        p90 = sorted(sp)[int(0.9 * (len(sp) - 1))] if sp else 0.0
        part = Segment("rotation", sub[0].t, sub[-1].t, sub[0].x, sub[0].y,
                       sub[-1].x, sub[-1].y, dist, dist / dur, max(sp) if sp else 0.0,
                       mode="vehicle" if p90 >= c["vehicle_speed_mps"] else "foot",
                       displacement_m=math.dist((sub[0].x, sub[0].y), (sub[-1].x, sub[-1].y)))
        parts.append(part)
    return parts or [seg]


def phases(zone: Sequence[ZoneSample], min_shrink: float = 0.08,
           confirm: int = 2) -> List[Phase]:
    """Phase announcements from the safe circle, closes from the blue circle.

    A phase is only declared once the new circle has been seen on ``confirm``
    consecutive samples: fitting noise otherwise invents phases that never
    happened.
    """
    safe = _median_filter([z for z in zone if z.ring == "safe"])
    blue = [z for z in zone if z.ring == "blue"]
    out: List[Phase] = []
    pending: List[ZoneSample] = []
    for z in safe:
        if not out:
            out.append(Phase(1, z.t, z.cx, z.cy, z.r))
            continue
        last = out[-1]
        moved = math.dist((z.cx, z.cy), (last.cx, last.cy)) > 0.25 * max(last.r, 1.0)
        shrank = z.r < last.r * (1.0 - min_shrink)
        if not (shrank or moved):
            pending = []
            continue
        if pending and (math.dist((z.cx, z.cy), (pending[-1].cx, pending[-1].cy))
                        <= 0.15 * max(z.r, 1.0)):
            pending.append(z)
        else:
            pending = [z]
        if len(pending) >= confirm:
            first = pending[0]
            out.append(Phase(last.number + 1, first.t, z.cx, z.cy, z.r))
            pending = []
    for ph in out:
        for b in blue:
            if b.t >= ph.announced_t and b.r <= ph.r * 1.02:
                ph.closed_t = b.t
                break
    return out


def zone_relation(s: Sample, ph: Phase) -> Tuple[float, bool]:
    """Distance to the safe-circle edge (negative inside) and whether inside."""
    d = math.dist((s.x, s.y), (ph.cx, ph.cy)) - ph.r
    return d, d <= 0.0


def time_outside(samples: Sequence[Sample], ph_list: Sequence[Phase],
                 blue: Sequence[ZoneSample]) -> float:
    """Seconds spent outside the blue circle (the part that costs health)."""
    if not blue:
        return 0.0
    total = 0.0
    bi = 0
    blue_sorted = sorted(blue, key=lambda b: b.t)
    for a, b in zip(samples, samples[1:]):
        while bi + 1 < len(blue_sorted) and blue_sorted[bi + 1].t <= a.t:
            bi += 1
        z = blue_sorted[bi]
        if math.dist((a.x, a.y), (z.cx, z.cy)) > z.r:
            total += b.t - a.t
    return total


def edge_fraction(samples: Sequence[Sample], ph_list: Sequence[Phase],
                  band: float = 0.25) -> Optional[float]:
    """Share of time spent in the outer ring of the safe circle."""
    if not ph_list or len(samples) < 2:
        return None
    inside = 0.0
    edge = 0.0
    for a, b in zip(samples, samples[1:]):
        ph = _phase_at(ph_list, a.t)
        if ph is None or ph.r <= 0:
            continue
        d = math.dist((a.x, a.y), (ph.cx, ph.cy))
        if d <= ph.r:
            dt = b.t - a.t
            inside += dt
            if d >= ph.r * (1.0 - band):
                edge += dt
    return (edge / inside) if inside > 0 else None


def _phase_at(ph_list: Sequence[Phase], t: float) -> Optional[Phase]:
    cur = None
    for ph in ph_list:
        if ph.announced_t <= t:
            cur = ph
        else:
            break
    return cur


def zone_entry(samples: Sequence[Sample], ph: Phase) -> Optional[Tuple[float, float]]:
    """(time entered, seconds before the zone closed). None if never entered."""
    for a in samples:
        if a.t < ph.announced_t:
            continue
        if math.dist((a.x, a.y), (ph.cx, ph.cy)) <= ph.r:
            margin = (ph.closed_t - a.t) if ph.closed_t is not None else float("nan")
            return a.t, margin
    return None


def rotation_start_after_announce(segs: Sequence[Segment], ph: Phase) -> Optional[float]:
    """Seconds between the phase being announced and the next rotation starting."""
    for s in segs:
        if s.kind == "rotation" and s.t0 >= ph.announced_t:
            return s.t0 - ph.announced_t
    return None


def squad_spread(by_player: Dict[str, List[Sample]], t: float,
                 tol: float = 0.6) -> Optional[float]:
    pts = []
    for samples in by_player.values():
        near = [s for s in samples if abs(s.t - t) <= tol]
        if near:
            s = min(near, key=lambda s: abs(s.t - t))
            pts.append((s.x, s.y))
    if len(pts) < 2:
        return None
    return max(math.dist(a, b) for i, a in enumerate(pts) for b in pts[i + 1:])
