"""Positional facts for a team, read from the phase-3 logs.

This is the module that closes the first line of phase 2's "not measurable
yet": positioning, rotations, zone-edge play and vehicle time all become
measurable the moment a map stream is analysed, and all four are read here
from map_events.jsonl and tracks.jsonl rather than estimated.

Definitions are the phase-3 ones and are printed with the numbers, because a
coach should be able to argue with them:

* rotation   -- a stretch over 3 m/s covering at least 200 m of ground and
                ending at least 150 m from where it started
* vehicle    -- a rotation whose 90th-percentile speed is over 11 m/s; the
                split is a speed threshold, not a sighting of a vehicle
* zone edge  -- time inside the safe circle spent in its outer quarter
* delay      -- seconds between the circle being announced and the rotation
                starting; margin is seconds between arriving and it closing
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .events import read_jsonl


def fmt_ts(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


@dataclass
class PositionalMatch:
    """Everything the map stream says about one team in one match."""

    number: int
    team: str
    source: str
    map_name: Optional[str] = None
    landing: Optional[Dict[str, Any]] = None
    loot_s: float = 0.0
    rotations: List[Dict[str, Any]] = field(default_factory=list)
    phases: List[Dict[str, Any]] = field(default_factory=list)
    zone_entries: List[Dict[str, Any]] = field(default_factory=list)
    edge_fraction: Optional[float] = None
    time_outside_blue_s: Optional[float] = None
    tracked_from_s: Optional[float] = None
    tracked_to_s: Optional[float] = None
    positions: List[Dict[str, Any]] = field(default_factory=list)
    world_size_m: float = 8000.0

    # ---- derived ----
    @staticmethod
    def _duration(r: Dict[str, Any]) -> float:
        """A log that predates the duration field still has its two timestamps."""
        if r.get("duration_s") is not None:
            return float(r["duration_s"])
        return max(0.0, float(r.get("t_end", 0.0)) - float(r.get("t", 0.0)))

    @property
    def vehicle_time_s(self) -> float:
        return sum(self._duration(r) for r in self.rotations if r.get("mode") == "vehicle")

    @property
    def foot_time_s(self) -> float:
        return sum(self._duration(r) for r in self.rotations if r.get("mode") == "foot")

    @property
    def moving_time_s(self) -> float:
        return self.vehicle_time_s + self.foot_time_s

    @property
    def vehicle_share(self) -> Optional[float]:
        return (self.vehicle_time_s / self.moving_time_s) if self.moving_time_s else None

    @property
    def rotation_distance_m(self) -> float:
        return sum(float(r.get("distance_m", 0)) for r in self.rotations)

    @property
    def delays(self) -> List[float]:
        return [float(r["after_announce_s"]) for r in self.rotations
                if r.get("after_announce_s") is not None]

    @property
    def margins(self) -> List[float]:
        return [float(r["before_close_s"]) for r in self.rotations
                if r.get("before_close_s") is not None]

    def cite(self, t: float, place: Optional[str] = None) -> str:
        return f"M{self.number} {fmt_ts(t)}" + (f" {place}" if place else "")


def _grid_ref(x: float, y: float, world: float) -> str:
    step = world / 8.0
    gx = min(7, max(0, int(x // step)))
    gy = min(7, max(0, int(y // step)))
    return "ABCDEFGH"[gx] + str(gy + 1)


def _team_pos(samples: Sequence[Dict[str, Any]], t: float,
              tol: float = 1.5) -> Optional[Tuple[float, float]]:
    rows = [s for s in samples if abs(float(s["t"]) - t) <= tol]
    if not rows:
        return None
    xs = sorted(float(r["x_m"]) for r in rows)
    ys = sorted(float(r["y_m"]) for r in rows)
    return xs[len(xs) // 2], ys[len(ys) // 2]


def positional_match(number: int, tracks_path: Optional[str],
                     map_events_path: Optional[str], team: str,
                     confidence_threshold: float = 0.6) -> Optional[PositionalMatch]:
    """Read one match's positional facts. None when no map stream was supplied."""
    if not map_events_path or not Path(map_events_path).exists():
        return None
    evs = [e for e in read_jsonl(map_events_path)
           if float(e.get("confidence", 1.0)) >= confidence_threshold]
    ours = [e for e in evs if e.get("team") == team and e.get("unit") == "team"]
    pm = PositionalMatch(number=number, team=team, source=Path(map_events_path).name)
    pm.phases = sorted(
        [{"n": int(e["phase"]), "t": float(e["t"]), "closed": e.get("closed_t"),
          "cx": float(e["cx_m"]), "cy": float(e["cy_m"]), "r": float(e["r_m"]),
          "place": e.get("place")}
         for e in evs if e.get("type") == "zone_phase"],
        key=lambda p: p["n"])
    land = next((e for e in ours if e["type"] == "landing"), None)
    if land is not None:
        pm.landing = {"t": float(land["t"]), "place": land.get("place"),
                      "x": float(land["x_m"]), "y": float(land["y_m"]),
                      "sure": land.get("state") != "no_descent_seen"}
    pm.rotations = sorted([e for e in ours if e["type"] == "rotation"],
                          key=lambda e: float(e["t"]))
    first_move = float(pm.rotations[0]["t"]) if pm.rotations else None
    pm.loot_s = sum(PositionalMatch._duration(e) for e in ours if e["type"] == "loot"
                    and (first_move is None or float(e["t"]) < first_move))
    zs = next((e for e in ours if e["type"] == "zone_summary"), None)
    if zs is not None:
        pm.edge_fraction = zs.get("edge_time_fraction")
        pm.time_outside_blue_s = zs.get("time_outside_blue_s")
        pm.tracked_from_s = zs.get("tracked_from_s")
        pm.tracked_to_s = zs.get("tracked_to_s")
    pm.zone_entries = sorted([e for e in ours if e["type"] == "zone_entry"],
                             key=lambda e: int(e.get("phase", 0)))

    if tracks_path and Path(tracks_path).exists():
        recs = read_jsonl(tracks_path)
        info = next((r for r in recs if r.get("type") == "map_info"), {})
        pm.world_size_m = float(info.get("world_size_m", 8000.0))
        pm.map_name = info.get("map")
        samples = [r for r in recs if r.get("type") == "sample" and r.get("team") == team
                   and float(r.get("confidence", 0)) >= confidence_threshold]
        for ph in pm.phases:
            p = _team_pos(samples, ph["t"])
            if p is None:
                pm.positions.append({"phase": ph["n"], "t": ph["t"], "place": None,
                                     "edge_m": None})
                continue
            edge = math.dist(p, (ph["cx"], ph["cy"])) - ph["r"]
            pm.positions.append({
                "phase": ph["n"], "t": ph["t"],
                "place": _grid_ref(p[0], p[1], pm.world_size_m),
                "edge_m": round(edge)})
    return pm


def positional_matches(numbers: Sequence[int], tracks_paths: Sequence[str],
                       map_events_paths: Sequence[str], team: str,
                       confidence_threshold: float = 0.6) -> List[PositionalMatch]:
    out: List[PositionalMatch] = []
    for i, n in enumerate(numbers):
        pm = positional_match(
            n,
            tracks_paths[i] if i < len(tracks_paths) else None,
            map_events_paths[i] if i < len(map_events_paths) else None,
            team, confidence_threshold)
        if pm is not None:
            out.append(pm)
    return out


# ---------------- rendering ----------------
def render_section(pms: Sequence[PositionalMatch], heading: str) -> List[str]:
    """The positioning section of a coach report."""
    L: List[str] = [heading, ""]
    if not pms:
        L.append("No map stream was supplied for this block, so nothing here can be "
                 "derived. Run `mapanalyze` on the map recordings and pass "
                 "`--tracks`/`--map-events`.")
        L.append("")
        return L
    L.append("A rotation is a stretch over 3 m/s covering at least 200 m and ending at "
             "least 150 m from where it started. Vehicle versus foot is a speed "
             "threshold (90th percentile over 11 m/s), not a sighting of a vehicle. "
             "Zone-edge time is time inside the safe circle spent in its outer quarter.")
    L.append("")
    L.append("| match | landed | looted there | rotations | distance moved | vehicle time | on foot | zone-edge time | outside the blue |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for pm in pms:
        landed = "not in log"
        if pm.landing:
            landed = f"{fmt_ts(pm.landing['t'])} {pm.landing['place'] or ''}".strip()
            if not pm.landing["sure"]:
                landed += " (no descent seen)"
        edge = ("not in log" if pm.edge_fraction is None
                else f"{float(pm.edge_fraction) * 100:.0f}%")
        outside = ("not in log" if pm.time_outside_blue_s is None
                   else f"{float(pm.time_outside_blue_s):.0f} s")
        L.append(f"| M{pm.number} | {landed} | {pm.loot_s:.0f} s | {len(pm.rotations)} | "
                 f"{pm.rotation_distance_m:.0f} m | {pm.vehicle_time_s:.0f} s | "
                 f"{pm.foot_time_s:.0f} s | {edge} | {outside} |")
    L.append("")

    carried = sum(1 for pm in pms for r in pm.rotations if r.get("continued"))
    total_rot = sum(len(pm.rotations) for pm in pms)
    if carried:
        L.append(f"{carried} of the {total_rot} are one movement continuing through an "
                 f"announcement; each part is shown against the circle it was moving "
                 f"towards, so a delay is only quoted where a rotation actually started.")
        L.append("")
    all_delays = [d for pm in pms for d in pm.delays]
    if all_delays:
        L.append(f"Rotation delay after the circle is announced: median "
                 f"{statistics.median(all_delays):.0f} s over {len(all_delays)} rotations "
                 f"({min(all_delays):.0f} s fastest, {max(all_delays):.0f} s slowest).")
        L.append("")
    veh = sum(pm.vehicle_time_s for pm in pms)
    moving = sum(pm.moving_time_s for pm in pms)
    if moving:
        L.append(f"Vehicle time: {veh:.0f} s of {moving:.0f} s moving "
                 f"({veh / moving * 100:.0f}% of time in rotations).")
        L.append("")

    L.append("Rotations, by zone phase:")
    L.append("")
    L.append("| match | phase | announced | we started | delay | mode | from → to | distance | arrived | margin to close |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for pm in pms:
        for ph in pm.phases:
            rots = [r for r in pm.rotations if int(r.get("phase", 0)) == ph["n"]]
            if not rots:
                L.append(f"| M{pm.number} | {ph['n']} | {fmt_ts(ph['t'])} | "
                         f"no rotation in log | - | - | - | - | - | - |")
                continue
            r = rots[0]
            margin = r.get("before_close_s")
            if r.get("ran_into_next_phase"):
                margin_txt = "still moving at the next announcement"
            elif margin is None:
                margin_txt = "-"
            elif float(margin) < 0:
                margin_txt = f"{-float(margin):.0f} s after it closed"
            else:
                margin_txt = f"{float(margin):.0f} s"
            delay_txt = ("already moving" if r.get("continued")
                         else f"{float(r.get('after_announce_s', 0)):.0f} s")
            L.append(f"| M{pm.number} | {ph['n']} | {fmt_ts(ph['t'])} | "
                     f"{fmt_ts(float(r['t']))} | "
                     f"{delay_txt} | {r.get('mode', '?')} | "
                     f"{r.get('from')} → {r.get('to')} | {float(r['distance_m']):.0f} m | "
                     f"{fmt_ts(float(r['t_end']))} | {margin_txt} |")
    L.append("")

    L.append("Where we were when each circle was announced:")
    L.append("")
    L.append("| match | phase | our place | to the circle edge |")
    L.append("|---|---|---|---|")
    any_pos = False
    for pm in pms:
        for p in pm.positions:
            any_pos = True
            edge = ("not tracked" if p["edge_m"] is None
                    else (f"{-int(p['edge_m'])} m inside" if p["edge_m"] <= 0
                          else f"{int(p['edge_m'])} m outside"))
            L.append(f"| M{pm.number} | {p['phase']} | {p['place'] or 'not tracked'} | {edge} |")
    if not any_pos:
        L.append("| - | - | no tracks file supplied | - |")
    L.append("")
    never = [(pm, e) for pm in pms for e in pm.zone_entries
             if e.get("state") == "never_entered"]
    if never:
        L.append("Phases we never entered: " + "; ".join(
            f"M{pm.number} phase {e.get('phase')}" for pm, e in never) + ".")
        L.append("")
    return L


def focus_candidates(pms: Sequence[PositionalMatch]) -> List[Tuple[float, Tuple[str, str, str]]]:
    """Practice focuses that only positional data can support."""
    out: List[Tuple[float, Tuple[str, str, str]]] = []
    delays = [(d, pm, r) for pm in pms for r, d in
              ((r, float(r["after_announce_s"])) for r in pm.rotations
               if r.get("after_announce_s") is not None)]
    if delays:
        med = statistics.median(d for d, _pm, _r in delays)
        worst = max(delays, key=lambda x: x[0])
        out.append((min(1.0, med / 60.0), (
            "Rotate earlier after the circle is announced",
            f"median delay {med:.0f}s across {len(delays)} tracked rotations; slowest "
            f"{worst[0]:.0f}s at {worst[1].cite(float(worst[2]['t']), worst[2].get('from'))}",
            f"median delay under {max(5, med * 0.6):.0f}s")))
    margins = [(float(r["before_close_s"]), pm, r) for pm in pms for r in pm.rotations
               if r.get("before_close_s") is not None]
    tight = [x for x in margins if x[0] < 30]
    if margins and tight:
        w = min(tight, key=lambda x: x[0])
        when = ("arrived %.0fs after it closed" % -w[0]) if w[0] < 0 else ("%.0fs to spare" % w[0])
        out.append((len(tight) / len(margins), (
            "Stop arriving on the closing edge",
            f"{len(tight)} of {len(margins)} rotations finished under 30s before the circle "
            f"closed; worst at {w[1].cite(float(w[2]['t_end']), w[2].get('to'))}, {when}",
            "no rotation finishing under 30s before close")))
    outside = [(float(pm.time_outside_blue_s), pm) for pm in pms
               if pm.time_outside_blue_s is not None]
    if outside and sum(v for v, _ in outside) > 0:
        tot = sum(v for v, _ in outside)
        worst = max(outside, key=lambda x: x[0])
        out.append((min(1.0, tot / (60.0 * len(outside))), (
            "Less time in the blue",
            f"{tot:.0f}s outside the circle across {len(outside)} matches; worst "
            f"M{worst[1].number} with {worst[0]:.0f}s",
            f"under {max(10, tot / len(outside) * 0.5):.0f}s per match")))
    edges = [(float(pm.edge_fraction), pm) for pm in pms if pm.edge_fraction is not None]
    if edges:
        med = statistics.median(v for v, _ in edges)
        if med > 0.35:
            worst = max(edges, key=lambda x: x[0])
            out.append((min(1.0, med), (
                "Hold ground further inside the circle",
                f"median {med * 100:.0f}% of time inside the circle was spent in its outer "
                f"quarter; worst M{worst[1].number} at {worst[0] * 100:.0f}%",
                f"under {max(15, med * 100 * 0.7):.0f}% next block")))
    drops = [(math.dist((pm.landing["x"], pm.landing["y"]),
                        (pm.phases[0]["cx"], pm.phases[0]["cy"])), pm)
             for pm in pms if pm.landing and pm.phases]
    if drops:
        med = statistics.median(v for v, _ in drops)
        far = max(drops, key=lambda x: x[0])
        out.append((min(1.0, med / 3000.0), (
            "Drop closer to the first circle",
            f"median {med:.0f} m from the first circle centre; furthest {far[0]:.0f} m at "
            f"{far[1].cite(float(far[1].landing['t']), far[1].landing['place'])}",
            f"median under {max(500, med * 0.7):.0f} m")))
    return out
