"""Build a self-contained dashboard from the logs the three phases produce.

One HTML file, no server and no database, because the rest of this toolchain
has neither: a coach opens it from a USB stick in the practice room. Every
number on the page comes from a log record, and anything a log marks unknown
is drawn as unknown rather than rounded into a number.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .events import read_jsonl
from .report import (CONVERSION_WINDOW_S, DEFAULT_PLACEMENT_POINTS, KILL_POINTS,
                     TRADE_WINDOW_S, _collect, _dropped_counts, _focuses, _phase_at,
                     _split_matches)

PATH_STEP_S = 2.0          # map paths are thinned to this, to keep the file small


def _series(match, typ: str) -> List[List[float]]:
    out: List[List[float]] = []
    for e in match.of(typ):
        if e.get("state") == "unknown" or e.get("value") is None:
            continue
        out.append([round(float(e["t"]) - match.start_t, 2), int(e["value"])])
    return out


def _match_events(match, team: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in match.of("kill", "knock"):
        v = f"{e.get('victim_team')}#{e.get('victim_slot')}"
        k = (f"{e.get('killer_team')}#{e.get('killer_slot')}"
             if e.get("killer_team") else None)
        ours = ("loss" if e.get("victim_team") == team
                else "gain" if e.get("killer_team") == team else "other")
        out.append({"t": round(float(e["t"]) - match.start_t, 2), "kind": e["type"],
                    "victim": v, "killer": k, "side": ours,
                    "phase": _phase_at(match, float(e["t"])),
                    "conf": round(float(e.get("confidence", 0)), 2)})
    out.sort(key=lambda r: r["t"])
    return out


def _map_block(tracks_path: Optional[str], map_events_path: Optional[str],
               threshold: float) -> Optional[Dict[str, Any]]:
    if not tracks_path or not Path(tracks_path).exists():
        return None
    recs = read_jsonl(tracks_path)
    info = next((r for r in recs if r.get("type") == "map_info"), {})
    world = float(info.get("world_size_m", 8000.0))
    paths: Dict[str, List[List[float]]] = {}
    last: Dict[str, float] = {}
    for r in recs:
        if r.get("type") != "sample" or not r.get("team"):
            continue
        if float(r.get("confidence", 0)) < threshold:
            continue
        t = float(r["t"])
        team = r["team"]
        if team in last and t - last[team] < PATH_STEP_S:
            continue
        last[team] = t
        paths.setdefault(team, []).append([round(t, 1), round(float(r["x_m"])),
                                           round(float(r["y_m"]))])
    phases: List[Dict[str, Any]] = []
    landings: List[Dict[str, Any]] = []
    rotations: List[Dict[str, Any]] = []
    if map_events_path and Path(map_events_path).exists():
        for e in read_jsonl(map_events_path):
            if e["type"] == "zone_phase":
                phases.append({"n": int(e["phase"]), "t": float(e["t"]),
                               "closed": e.get("closed_t"),
                               "cx": round(float(e["cx_m"])), "cy": round(float(e["cy_m"])),
                               "r": round(float(e["r_m"])), "place": e.get("place")})
            elif e["type"] == "landing" and e.get("unit") == "team":
                landings.append({"team": e["team"], "t": round(float(e["t"]), 1),
                                 "x": round(float(e["x_m"])), "y": round(float(e["y_m"])),
                                 "place": e.get("place"),
                                 "sure": e.get("state") != "no_descent_seen"})
            elif e["type"] == "rotation" and e.get("unit") == "team":
                rotations.append({"team": e["team"], "t": round(float(e["t"]), 1),
                                  "t_end": round(float(e["t_end"]), 1),
                                  "from": e.get("from"), "to": e.get("to"),
                                  "distance": round(float(e.get("distance_m", 0))),
                                  "mode": e.get("mode"), "phase": e.get("phase"),
                                  "delay": e.get("after_announce_s"),
                                  "margin": e.get("before_close_s")})
    if not paths and not phases:
        return None
    end = max([p[0] for pts in paths.values() for p in pts] +
              [p["t"] for p in phases] + [0.0])
    return {"world": world, "name": info.get("map"), "paths": paths, "phases": phases,
            "landings": landings, "rotations": rotations, "duration": round(end, 1),
            "ourTeam": info.get("our_team")}


def build_model(events_paths: Sequence[str], team: str,
                tracks_paths: Optional[Sequence[str]] = None,
                map_events_paths: Optional[Sequence[str]] = None,
                block_name: str = "scrim block",
                placement_points: Optional[Dict[int, int]] = None,
                confidence_threshold: float = 0.6) -> Dict[str, Any]:
    pts = {int(k): int(v) for k, v in (placement_points or DEFAULT_PLACEMENT_POINTS).items()}
    matches = _split_matches(events_paths, confidence_threshold)
    facts = [_collect(m, team) for m in matches]
    tracks_paths = list(tracks_paths or [])
    map_events_paths = list(map_events_paths or [])

    out_matches: List[Dict[str, Any]] = []
    cum = 0
    for i, f in enumerate(facts):
        m = f.m
        placement_pts = pts.get(f.placement, 0) if f.placement is not None else 0
        total = placement_pts + f.kill_points
        cum += total
        mp = _map_block(tracks_paths[i] if i < len(tracks_paths) else None,
                        map_events_paths[i] if i < len(map_events_paths) else None,
                        confidence_threshold)
        zone_phases = [{"n": int(e["value"]), "t": round(float(e["t"]) - m.start_t, 2)}
                       for e in m.of("zone_phase") if e.get("value") is not None]
        alive = {s: round(v[0], 1) for s, v in f.alive_times.items()}
        out_matches.append({
            "n": m.number, "source": m.source, "map": f.map_name,
            "placement": f.placement, "placementPts": placement_pts,
            "kills": len(f.our_kills), "killPts": f.kill_points, "total": total,
            "cumulative": cum, "winner": f.winner, "fullWipe": f.full_wipe,
            "duration": round(m.end_t - m.start_t, 1),
            "teamsAlive": _series(m, "teams_alive"),
            "playersAlive": _series(m, "players_alive"),
            "zonePhases": zone_phases,
            "events": _match_events(m, team),
            "aliveTimes": alive,
            "map3": mp,
        })

    # ---- block-level tables, same definitions as the markdown report ----
    totals = [mm["total"] for mm in out_matches]
    avg = sum(totals) / len(totals) if totals else 0.0
    trades = chances = conv = conv_den = lost = lost_den = 0
    opponents: Dict[str, List[Dict[str, Any]]] = {}
    unattributed = 0
    phase_rows: Dict[Tuple[int, Any], List[int]] = {}
    pre3: List[Dict[str, Any]] = []
    for f in facts:
        m = f.m
        for e in f.our_knocked:
            if not e.get("killer_team"):
                continue
            chances += 1
            if [x for x in (f.our_knocks + f.our_kills)
                    if x.get("victim_team") == e["killer_team"]
                    and 0 <= float(x["t"]) - float(e["t"]) <= TRADE_WINDOW_S]:
                trades += 1
        for e in f.our_knocks:
            conv_den += 1
            if [x for x in m.of("kill")
                    if x.get("victim_team") == e.get("victim_team")
                    and x.get("victim_slot") == e.get("victim_slot")
                    and 0 <= float(x["t"]) - float(e["t"]) <= CONVERSION_WINDOW_S]:
                conv += 1
        for e in f.our_knocked:
            lost_den += 1
            if [x for x in f.our_deaths if x.get("victim_slot") == e.get("victim_slot")
                    and 0 <= float(x["t"]) - float(e["t"]) <= CONVERSION_WINDOW_S]:
                lost += 1
        for e in f.our_kills:
            phase_rows.setdefault((m.number, _phase_at(m, float(e["t"]))), [0, 0])[0] += 1
        for e in f.our_deaths:
            ph = _phase_at(m, float(e["t"]))
            phase_rows.setdefault((m.number, ph), [0, 0])[1] += 1
            if ph is not None and int(ph) < 3:
                pre3.append({"m": m.number, "t": round(float(e["t"]) - m.start_t, 1),
                             "slot": e.get("victim_slot"),
                             "killer": e.get("killer_team")})
            if e.get("killer_team"):
                opponents.setdefault(e["killer_team"], []).append(
                    {"m": m.number, "t": round(float(e["t"]) - m.start_t, 1), "phase": ph})
            else:
                unattributed += 1

    focuses = [{"title": t, "evidence": ev, "target": tg}
               for t, ev, tg in _focuses(facts, trades, chances, conv, conv_den,
                                         lost, lost_den,
                                         [(p["m"], {"t": p["t"], "victim_slot": p["slot"],
                                                    "killer_team": p["killer"]})
                                          for p in pre3])[:3]]
    quality = _dropped_counts(events_paths, confidence_threshold)
    from .report import NOT_MEASURABLE
    from .map_report import NOT_MEASURABLE as MAP_NOT_MEASURABLE

    return {
        "team": team, "block": block_name,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "threshold": confidence_threshold,
        "pointsTable": {str(k): v for k, v in sorted(pts.items())},
        "killPoints": KILL_POINTS,
        "matches": out_matches,
        "blockAverage": round(avg, 2),
        "survival": [{"m": f.m.number,
                      "avgAlive": round(sum(v[0] for v in f.alive_times.values())
                                        / max(1, len(f.alive_times)), 1),
                      "firstDeath": (round(min(float(e["t"]) for e in f.our_deaths)
                                           - f.m.start_t, 1) if f.our_deaths else None),
                      "lastDeath": (round(max(float(e["t"]) for e in f.our_deaths)
                                          - f.m.start_t, 1) if f.our_deaths else None),
                      "fullWipe": f.full_wipe,
                      "top5": (None if f.placement is None else f.placement <= 5)}
                     for f in facts],
        "phaseRows": [{"m": k[0], "phase": k[1], "kills": v[0], "deaths": v[1]}
                      for k, v in sorted(phase_rows.items(),
                                         key=lambda kv: (kv[0][0], kv[0][1] is None,
                                                         kv[0][1] or 0))],
        "fights": {"trades": trades, "tradeChances": chances,
                   "conversions": conv, "conversionDen": conv_den,
                   "knocksLost": lost, "knocksLostDen": lost_den,
                   "preZone3": pre3},
        "opponents": [{"team": k, "lives": len(v), "rows": v}
                      for k, v in sorted(opponents.items(), key=lambda kv: -len(kv[1]))],
        "unattributedDeaths": unattributed,
        "focuses": focuses,
        "quality": quality,
        "notMeasurable": list(NOT_MEASURABLE),
        "mapNotMeasurable": list(MAP_NOT_MEASURABLE),
    }


ARTIFACT_CSS = """
<style>
/* published-page variant: the host wraps the file, so the page keeps its own
   dark palette and clears a system bar at the phone's side */
:root{color-scheme:dark}
.masthead,.stage,.sheets,.foot{
  padding-left:max(var(--gut),env(safe-area-inset-left,0px));
  padding-right:max(var(--gut),env(safe-area-inset-right,0px));
}
.rail{padding-left:max(14px,env(safe-area-inset-left,0px))}
.readout{padding-right:max(16px,env(safe-area-inset-right,0px))}
</style>
"""


def render_html(model: Dict[str, Any], artifact: bool = False) -> str:
    tpl = (Path(__file__).parent / "dashboard_template.html").read_text()
    tpl = tpl.replace("<title>Scrim block</title>",
                      f"<title>{model['team']} {model['block']}</title>")
    if artifact:
        head = tpl[tpl.index("<title>"):tpl.index("</head>")]
        body = tpl[tpl.index("<body>") + len("<body>"):tpl.index("</body>")]
        tpl = ('<meta name="claude:viewport-fit-sides" content="cover">\n'
               + head + ARTIFACT_CSS + body)
    return tpl.replace("/*%%MODEL%%*/null",
                       json.dumps(model, separators=(",", ":"), allow_nan=False))


def write_dashboard(out_path: str, model: Dict[str, Any], artifact: bool = False) -> str:
    html = render_html(model, artifact=artifact)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html)
    return out_path
