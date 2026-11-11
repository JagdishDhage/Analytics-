"""Phase 3 report: what the map stream says about how a team played.

Built from tracks.jsonl and map_events.jsonl only, optionally fused with a
phase-1 events.jsonl so that kills and knocks can be placed on the map. The
same rule as phase 2 applies: nothing is derived that the logs do not
contain, every figure cites a time and a place, and anything the medium
cannot support is listed rather than estimated.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .events import read_jsonl
from .map_world import WorldMap

NOT_MEASURABLE = [
    "Elevation, buildings and whether a team is inside or outside: the map "
    "stream is a flat top-down view.",
    "Which player of a squad is which while their dots overlap: at the map's "
    "scale team-mates standing together are one dot, so positions are reported "
    "per team with the player named only where the stream showed a readable name.",
    "Vehicle type, fuel or whether a team is on foot in a vehicle's shadow: "
    "only speed is measurable, and the foot/vehicle split is a speed threshold.",
    "The zone timer itself: phase announcements and closes are read from the "
    "circles moving, not from the countdown.",
    "Damage, knocks and kills: those come from the HUD stream (phase 1). Pass "
    "an events log to place them on the map; without one this report has no fights.",
    "Loot quality and what a team picked up: looting here means a stretch of "
    "standing still, nothing more.",
]


def _fmt(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


@dataclass
class MapMatch:
    number: int
    source: str
    tracks: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    map_name: Optional[str] = None
    world: Optional[WorldMap] = None
    hud_events: List[Dict[str, Any]] = field(default_factory=list)
    time_offset: float = 0.0

    def of(self, *types: str) -> List[Dict[str, Any]]:
        return [e for e in self.events if e.get("type") in types]

    def team_events(self, team: str, *types: str) -> List[Dict[str, Any]]:
        return [e for e in self.of(*types)
                if e.get("team") == team and e.get("unit", "team") == "team"]

    def cite(self, t: float, place: Optional[str] = None) -> str:
        return f"M{self.number} {_fmt(t)}" + (f" {place}" if place else "")


def _load_match(n: int, tracks_path: str, events_path: Optional[str],
                hud_path: Optional[str], threshold: float,
                time_offset: float = 0.0) -> MapMatch:
    tracks = [r for r in read_jsonl(tracks_path)
              if r.get("type") != "sample" or float(r.get("confidence", 0)) >= threshold]
    events = []
    if events_path and Path(events_path).exists():
        events = [e for e in read_jsonl(events_path)
                  if float(e.get("confidence", 1.0)) >= threshold]
    info = next((r for r in tracks if r.get("type") == "map_info"), {})
    world = None
    if info:
        world = WorldMap((0, 0, 1, 1), float(info.get("world_size_m", 8000.0)),
                         info.get("map"))
    hud = []
    if hud_path and Path(hud_path).exists():
        hud = [e for e in read_jsonl(hud_path)
               if e.get("state", "ok") == "ok" and float(e.get("confidence", 0)) >= threshold]
    return MapMatch(n, Path(tracks_path).name, tracks, events, info.get("map"), world,
                    hud, time_offset)


def _samples_at(m: MapMatch, t: float, tol: float = 0.6) -> List[Dict[str, Any]]:
    return [r for r in m.tracks if r.get("type") == "sample" and abs(float(r["t"]) - t) <= tol]


def _team_pos(m: MapMatch, team: str, t: float, tol: float = 0.6
              ) -> Optional[Tuple[float, float]]:
    rows = [r for r in _samples_at(m, t, tol) if r.get("team") == team]
    if not rows:
        return None
    xs = sorted(float(r["x_m"]) for r in rows)
    ys = sorted(float(r["y_m"]) for r in rows)
    return xs[len(xs) // 2], ys[len(ys) // 2]


def _pressure(m: MapMatch, team: str, t: float, radius_m: float = 200.0
              ) -> Tuple[int, List[str]]:
    """How many other players' dots are near us, and which teams they are."""
    pos = _team_pos(m, team, t)
    if pos is None:
        return 0, []
    near = 0
    teams: List[str] = []
    for r in _samples_at(m, t):
        if r.get("team") == team:
            continue
        d = math.dist((float(r["x_m"]), float(r["y_m"])), pos)
        if d <= radius_m:
            near += int(r.get("cluster", 1))
            if r.get("team") and r["team"] not in teams:
                teams.append(r["team"])
    return near, teams


def build_map_report(tracks_paths: Sequence[str], events_paths: Sequence[str],
                     team: str, hud_events_paths: Optional[Sequence[str]] = None,
                     block_name: str = "scrim block", confidence_threshold: float = 0.6,
                     time_offsets: Optional[Sequence[float]] = None) -> str:
    hud_events_paths = list(hud_events_paths or [])
    time_offsets = list(time_offsets or [])
    matches: List[MapMatch] = []
    for i, tp in enumerate(tracks_paths):
        ep = events_paths[i] if i < len(events_paths) else None
        hp = hud_events_paths[i] if i < len(hud_events_paths) else None
        off = time_offsets[i] if i < len(time_offsets) else 0.0
        matches.append(_load_match(i + 1, tp, ep, hp, confidence_threshold, off))

    L: List[str] = [f"# {team} — {block_name} map report", ""]
    L.append(f"Matches: {len(matches)} "
             f"({'; '.join(f'M{m.number}={m.source}' for m in matches)}). "
             f"Positions read at confidence >= {confidence_threshold:.2f}. "
             f"Places are named POIs where a POI table was calibrated, and map grid "
             f"references otherwise.")
    L.append("")
    covered = [m for m in matches if any(r.get("team") == team for r in m.tracks
                                         if r.get("type") == "sample")]
    if not covered:
        L.append(f"No dot in these logs was identified as {team}. Nothing below can be "
                 f"derived; check the roster in the map profile and the label geometry "
                 f"with `mapcalibrate --preview`.")
        return "\n".join(L)

    # ---------------- 1. drops and loot ----------------
    L += ["## 1. Drop and loot", "",
          "| match | map | landed | place | first move | looted there | drop to first circle centre |",
          "|---|---|---|---|---|---|---|"]
    for m in matches:
        land = next((e for e in m.team_events(team, "landing")), None)
        loots = m.team_events(team, "loot")
        rots = m.team_events(team, "rotation")
        ph1 = next((e for e in m.of("zone_phase") if e.get("phase") == 1), None)
        if land is None:
            L.append(f"| M{m.number} | {m.map_name or '?'} | not in log | - | - | - | - |")
            continue
        first_move = rots[0]["t"] if rots else None
        loot_s = sum(float(e["duration_s"]) for e in loots
                     if float(e["t"]) < (first_move if first_move is not None else 1e9))
        dist_txt = "-"
        if ph1:
            dist_txt = (f"{math.dist((float(land['x_m']), float(land['y_m'])), (float(ph1['cx_m']), float(ph1['cy_m']))):.0f} m")
        L.append(f"| M{m.number} | {m.map_name or '?'} | {_fmt(float(land['t']))} | "
                 f"{land['place']} | "
                 f"{_fmt(first_move) if first_move is not None else 'no rotation in log'} | "
                 f"{loot_s:.0f} s | {dist_txt} |")
    L.append("")

    # ---------------- 2. rotations ----------------
    L += ["## 2. Rotations, by zone phase", "",
          "| match | phase | announced | we started | delay | mode | from → to | distance | arrived | margin to close |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    late_rows: List[str] = []
    for m in matches:
        phases = {int(e["phase"]): e for e in m.of("zone_phase") if e.get("phase")}
        for ph_no in sorted(phases):
            ph = phases[ph_no]
            rots = [e for e in m.team_events(team, "rotation")
                    if int(e.get("phase", 0)) == ph_no]
            if not rots:
                L.append(f"| M{m.number} | {ph_no} | {_fmt(float(ph['t']))} | "
                         f"no rotation in log | - | - | - | - | - | - |")
                continue
            r = rots[0]
            margin = r.get("before_close_s")
            L.append(f"| M{m.number} | {ph_no} | {_fmt(float(ph['t']))} | "
                     f"{_fmt(float(r['t']))} | {float(r.get('after_announce_s', 0)):.0f} s | "
                     f"{r.get('mode', '?')} | {r['from']} → {r['to']} | "
                     f"{float(r['distance_m']):.0f} m | {_fmt(float(r['t_end']))} | "
                     f"{('%.0f s' % float(margin)) if margin is not None else '-'} |")
            if margin is not None and float(margin) < 30:
                late_rows.append(f"{m.cite(float(r['t_end']), r['to'])} phase {ph_no}, "
                                 f"{float(margin):.0f}s before the circle closed")
    L.append("")
    if late_rows:
        L.append("Arrived with under 30 s to spare: " + "; ".join(late_rows) + ".")
    else:
        L.append("No phase was entered with under 30 s to spare in these logs.")
    L.append("")

    # ---------------- 3. zone discipline ----------------
    L += ["## 3. Zone discipline", "",
          "| match | time outside the blue circle | share of time on the circle edge | phases never entered | tracked window |",
          "|---|---|---|---|---|"]
    for m in matches:
        zs = next((e for e in m.team_events(team, "zone_summary")), None)
        never = [str(e.get("phase")) for e in m.team_events(team, "zone_entry")
                 if e.get("state") == "never_entered"]
        if zs is None:
            L.append(f"| M{m.number} | not in log | - | - | - |")
            continue
        edge = zs.get("edge_time_fraction")
        L.append(f"| M{m.number} | {float(zs['time_outside_blue_s']):.0f} s | "
                 f"{edge if edge is not None else '-'} | "
                 f"{', '.join(never) if never else 'none'} | "
                 f"{_fmt(float(zs['tracked_from_s']))}–{_fmt(float(zs['tracked_to_s']))} |")
    L.append("")

    # ---------------- 4. fights on the map ----------------
    L += ["## 4. Fights, placed on the map", ""]
    any_hud = any(m.hud_events for m in matches)
    if not any_hud:
        L.append("No HUD event log was supplied, so no kill or knock can be placed. "
                 "Pass `--hud-events` with the phase-1 events.jsonl for the same match.")
    else:
        L += ["| match | t | event | opponent | our place | players within 200 m | teams nearby |",
              "|---|---|---|---|---|---|---|"]
        for m in matches:
            for e in sorted(m.hud_events, key=lambda e: float(e["t"])):
                if e.get("type") not in ("kill", "knock"):
                    continue
                ours_victim = e.get("victim_team") == team
                ours_killer = e.get("killer_team") == team
                if not (ours_victim or ours_killer):
                    continue
                t = float(e["t"]) + m.time_offset
                pos = _team_pos(m, team, t)
                place = m.world.place(*pos) if (m.world and pos) else "not tracked"
                near, teams_near = _pressure(m, team, t)
                opp = e.get("killer_team") if ours_victim else e.get("victim_team")
                label = ("death" if (ours_victim and e["type"] == "kill")
                         else "knocked" if ours_victim
                         else "kill" if e["type"] == "kill" else "knock")
                L.append(f"| M{m.number} | {_fmt(t)} | {label} | {opp or 'unattributed'} | "
                         f"{place} | {near} | {', '.join(teams_near) or 'none identified'} |")
    L.append("")

    # ---------------- 5. who was around us ----------------
    L += ["## 5. Pressure at phase changes", "",
          "| match | phase | our place | players within 200 m | teams identified nearby |",
          "|---|---|---|---|---|"]
    for m in matches:
        for ph in sorted(m.of("zone_phase"), key=lambda e: float(e["t"])):
            t = float(ph["t"])
            pos = _team_pos(m, team, t)
            if pos is None:
                continue
            near, teams_near = _pressure(m, team, t)
            L.append(f"| M{m.number} | {ph.get('phase')} | "
                     f"{m.world.place(*pos) if m.world else '-'} | {near} | "
                     f"{', '.join(teams_near) or 'none identified'} |")
    L.append("")

    # ---------------- 6. cohesion ----------------
    L += ["## 6. Squad cohesion", ""]
    rows = []
    for m in matches:
        sp = [float(e["value_m"]) for e in m.of("squad_spread") if e.get("team") == team]
        if not sp:
            rows.append(f"| M{m.number} | not measurable (no two named team-mates tracked "
                        f"at the same time) | - | - |")
            continue
        rows.append(f"| M{m.number} | {statistics.median(sp):.0f} m | {max(sp):.0f} m | "
                    f"{sum(1 for v in sp if v > 250) / len(sp):.2f} |")
    L += ["| match | median spread | widest | share of samples over 250 m |",
          "|---|---|---|---|"] + rows
    L.append("")
    L.append("Spread is measured between named team-mates only. While their dots overlap "
             "the stream cannot separate them, which is itself the measurement that the "
             "squad was together.")
    L.append("")

    # ---------------- 7. focuses ----------------
    L += ["## 7. Practice focuses", ""]
    focuses = _map_focuses(matches, team)
    for i, (title, evidence, target) in enumerate(focuses[:3], start=1):
        L.append(f"{i}. **{title}**")
        L.append(f"   - Evidence: {evidence}")
        L.append(f"   - Target next block: {target}")
    if not focuses:
        L.append("The logs hold too little tracked time for this team to rank focuses.")
    L.append("")

    # ---------------- 8. every other team the stream identified ----------------
    L += ["## 8. Other teams seen on the map", "",
          "Every team the stream named, not just ours: where they landed, how often they "
          "moved, and when the stream last had them. This is scouting material; a team "
          "the stream never named does not appear.", ""]
    for m in matches:
        others = sorted({e["team"] for e in m.of("landing", "rotation", "zone_summary")
                         if e.get("unit") == "team" and e.get("team") != team})
        if not others:
            L.append(f"M{m.number}: no other team identified.")
            L.append("")
            continue
        L += [f"**M{m.number}** ({m.map_name or '?'})", "",
              "| team | landed | place | rotations | first rotation | last seen |",
              "|---|---|---|---|---|---|"]
        for opp in others:
            land = next((e for e in m.of("landing")
                         if e.get("team") == opp and e.get("unit") == "team"), None)
            rots = [e for e in m.of("rotation")
                    if e.get("team") == opp and e.get("unit") == "team"]
            zs = next((e for e in m.of("zone_summary")
                       if e.get("team") == opp and e.get("unit") == "team"), None)
            first = rots[0] if rots else None
            L.append(f"| {opp} | {_fmt(float(land['t'])) if land else '-'} | "
                     f"{land['place'] if land else '-'} | {len(rots)} | "
                     + (f"{_fmt(float(first['t']))} {first['from']} → {first['to']} "
                        f"({first.get('mode', '?')})" if first else "none in log")
                     + f" | {_fmt(float(zs['tracked_to_s'])) if zs else '-'} |")
        L.append("")

    # ---------------- 9. limits ----------------
    L += ["## 9. Not measurable from a map stream", ""]
    for item in NOT_MEASURABLE:
        L.append(f"- {item}")
    L.append("")
    return "\n".join(L)


def _map_focuses(matches: Sequence[MapMatch], team: str) -> List[Tuple[str, str, str]]:
    out: List[Tuple[float, Tuple[str, str, str]]] = []
    delays: List[Tuple[float, MapMatch, Dict[str, Any]]] = []
    margins: List[Tuple[float, MapMatch, Dict[str, Any]]] = []
    outside: List[Tuple[float, MapMatch]] = []
    for m in matches:
        for r in m.team_events(team, "rotation"):
            if r.get("after_announce_s") is not None:
                delays.append((float(r["after_announce_s"]), m, r))
            if r.get("before_close_s") is not None:
                margins.append((float(r["before_close_s"]), m, r))
        zs = next((e for e in m.team_events(team, "zone_summary")), None)
        if zs is not None:
            outside.append((float(zs["time_outside_blue_s"]), m))
    if delays:
        med = statistics.median(d for d, _m, _r in delays)
        worst = max(delays, key=lambda d: d[0])
        out.append((min(1.0, med / 60.0), (
            "Rotate earlier after the circle is announced",
            f"median delay {med:.0f}s across {len(delays)} tracked rotations; slowest "
            f"{worst[0]:.0f}s at {worst[1].cite(float(worst[2]['t']), worst[2]['from'])}",
            f"median delay under {max(5, med * 0.6):.0f}s")))
    if margins:
        tight = [x for x in margins if x[0] < 30]
        if tight:
            w = min(tight, key=lambda x: x[0])
            out.append((min(1.0, len(tight) / max(1, len(margins))), (
                "Stop arriving on the closing edge",
                f"{len(tight)} of {len(margins)} rotations finished under 30s before the "
                f"circle closed; tightest {w[0]:.0f}s at "
                f"{w[1].cite(float(w[2]['t_end']), w[2]['to'])}",
                f"no rotation finishing under 30s before close")))
    if outside:
        tot = sum(v for v, _m in outside)
        if tot > 0:
            worst = max(outside, key=lambda x: x[0])
            out.append((min(1.0, tot / (60.0 * len(outside))), (
                "Less time in the blue",
                f"{tot:.0f}s outside the circle across {len(outside)} matches; worst "
                f"M{worst[1].number} with {worst[0]:.0f}s",
                f"under {max(10, tot / len(outside) * 0.5):.0f}s per match")))
    drops = []
    for m in matches:
        land = next((e for e in m.team_events(team, "landing")), None)
        ph1 = next((e for e in m.of("zone_phase") if e.get("phase") == 1), None)
        if land and ph1:
            drops.append((math.dist((float(land["x_m"]), float(land["y_m"])),
                                    (float(ph1["cx_m"]), float(ph1["cy_m"]))), m, land))
    if drops:
        med = statistics.median(d for d, _m, _l in drops)
        far = max(drops, key=lambda d: d[0])
        out.append((min(1.0, med / 3000.0), (
            "Drop closer to the first circle",
            f"median {med:.0f} m from the first circle centre; furthest {far[0]:.0f} m at "
            f"{far[1].cite(float(far[2]['t']), far[2]['place'])}",
            f"median under {max(500, med * 0.7):.0f} m")))
    out.sort(key=lambda x: -x[0])
    return [o[1] for o in out]
