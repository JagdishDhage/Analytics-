"""Ground-truth harness for the map stream (phase 3).

Scores extracted tracks against hand-labelled (or simulated) positions and
reports, in metres: how far the dots land from the truth, how often an
identity is right, how much of the lobby is seen at all, how well the zone
circles are fitted, and whether the behaviour layer (landings, rotations,
phases) built on top of the tracks agrees with the same layer built on the
truth. Label format (JSONL):

  {"kind":"pos","t":62.0,"team":"TX","player":"ALFA","x_m":4120.0,"y_m":3980.0}
  {"kind":"zone","t":62.0,"ring":"safe","cx_m":4000,"cy_m":4000,"r_m":1500}
  {"kind":"map_info","map":"ERANGEL","world_size_m":8000}
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .events import read_jsonl
from .map_analyze import derive_events
from .map_profile import MapProfile


def _nearest(samples: Sequence[Dict[str, Any]], t: float, tol: float) -> Optional[Dict[str, Any]]:
    best, best_dt = None, tol
    for s in samples:
        dt = abs(float(s["t"]) - t)
        if dt <= best_dt:
            best, best_dt = s, dt
    return best


@dataclass
class MapReport:
    tolerance_s: float = 0.35
    identity_tol_m: float = 80.0
    position: Dict[str, Any] = field(default_factory=dict)
    team_position: Dict[str, Any] = field(default_factory=dict)
    identity: Dict[str, Any] = field(default_factory=dict)
    coverage: Dict[str, Any] = field(default_factory=dict)
    zones: Dict[str, Any] = field(default_factory=dict)
    behaviour: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"tolerance_s": self.tolerance_s, "identity_tol_m": self.identity_tol_m,
                "position": self.position, "team_position": self.team_position,
                "identity": self.identity, "coverage": self.coverage,
                "zones": self.zones, "behaviour": self.behaviour, "notes": self.notes}


def _stats(vals: Sequence[float]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0}
    sv = sorted(vals)
    return {"n": len(sv), "median_m": round(statistics.median(sv), 2),
            "mean_m": round(sum(sv) / len(sv), 2),
            "p90_m": round(sv[int(0.9 * (len(sv) - 1))], 2),
            "max_m": round(sv[-1], 2),
            "within_25m": round(sum(1 for v in sv if v <= 25) / len(sv), 4),
            "within_50m": round(sum(1 for v in sv if v <= 50) / len(sv), 4),
            "within_100m": round(sum(1 for v in sv if v <= 100) / len(sv), 4)}


def truth_records(labels: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn labels into the same shape as tracks.jsonl, for a like-for-like
    comparison of everything derived on top."""
    out: List[Dict[str, Any]] = []
    tracks: Dict[Tuple[str, str], int] = {}
    for l in labels:
        if l.get("kind") == "pos":
            key = (l["team"], l["player"])
            tracks.setdefault(key, len(tracks) + 1)
            out.append({"type": "sample", "t": float(l["t"]), "track": tracks[key],
                        "team": l["team"], "player": l["player"],
                        "x_m": float(l["x_m"]), "y_m": float(l["y_m"]),
                        "cluster": 1, "confidence": 1.0, "place": ""})
        elif l.get("kind") == "zone":
            out.append({"type": "zone", "t": float(l["t"]), "ring": l["ring"],
                        "cx_m": float(l["cx_m"]), "cy_m": float(l["cy_m"]),
                        "r_m": float(l["r_m"]), "confidence": 1.0})
    return out


def score(labels_path: str | Path, tracks_path: str | Path, prof: MapProfile,
          tolerance: float = 0.35, identity_tol_m: float = 80.0) -> MapReport:
    labels = read_jsonl(labels_path)
    tracks = read_jsonl(tracks_path)
    rep = MapReport(tolerance_s=tolerance, identity_tol_m=identity_tol_m)

    truth_pos: Dict[Tuple[float, str, str], Tuple[float, float]] = {}
    truth_by_t: Dict[float, List[Tuple[str, str, float, float]]] = {}
    for l in labels:
        if l.get("kind") != "pos":
            continue
        t = round(float(l["t"]), 3)
        truth_pos[(t, l["team"], l["player"])] = (float(l["x_m"]), float(l["y_m"]))
        truth_by_t.setdefault(t, []).append((l["team"], l["player"],
                                             float(l["x_m"]), float(l["y_m"])))
    samples = [r for r in tracks if r.get("type") == "sample"]
    by_t: Dict[float, List[Dict[str, Any]]] = {}
    for s in samples:
        by_t.setdefault(round(float(s["t"]), 3), []).append(s)

    # ---- per-player position error, where a player identity was emitted ----
    perrs: List[float] = []
    pid_ok = pid_bad = 0
    for (t, team, player), (tx, ty) in truth_pos.items():
        cands = [s for s in by_t.get(t, []) if s.get("team") == team
                 and s.get("player") == player]
        if not cands:
            continue
        s = cands[0]
        e = math.dist((float(s["x_m"]), float(s["y_m"])), (tx, ty))
        perrs.append(e)
        if e <= identity_tol_m:
            pid_ok += 1
        else:
            pid_bad += 1
    rep.position = _stats(perrs)

    # ---- team-level: is the dot near *some* player of the team it claims? ----
    tid_ok = tid_bad = 0
    terrs: List[float] = []
    for t, srows in by_t.items():
        truth_rows = truth_by_t.get(t, [])
        for s in srows:
            team = s.get("team")
            if not team:
                continue
            mates = [(x, y) for (tm, _p, x, y) in truth_rows if tm == team]
            if not mates:
                tid_bad += 1
                continue
            d = min(math.dist((float(s["x_m"]), float(s["y_m"])), m) for m in mates)
            terrs.append(d)
            if d <= identity_tol_m:
                tid_ok += 1
            else:
                tid_bad += 1
    rep.team_position = _stats(terrs)
    named = [s for s in samples if s.get("team")]
    with_player = [s for s in samples if s.get("player")]
    rep.identity = {
        "samples": len(samples),
        "with_team": len(named),
        "with_player": len(with_player),
        "team_correct": tid_ok, "team_wrong": tid_bad,
        "team_precision": round(tid_ok / (tid_ok + tid_bad), 4) if (tid_ok + tid_bad) else None,
        "player_correct": pid_ok, "player_wrong": pid_bad,
        "player_precision": round(pid_ok / (pid_ok + pid_bad), 4) if (pid_ok + pid_bad) else None,
    }

    # ---- coverage: how much of the lobby is visible at all ----
    seen_players = cov_players = 0
    seen_teams = cov_teams = 0
    for t, rows in truth_by_t.items():
        srows = by_t.get(t, [])
        slots = sum(max(1, int(s.get("cluster", 1))) for s in srows)
        cov_players += len(rows)
        seen_players += min(len(rows), slots)
        teams = {tm for (tm, _p, _x, _y) in rows}
        cov_teams += len(teams)
        seen_teams += len({s.get("team") for s in srows if s.get("team")} & teams)
    rep.coverage = {
        "player_slots_truth": cov_players,
        "player_slots_covered_by_dots": seen_players,
        "dot_coverage": round(seen_players / cov_players, 4) if cov_players else None,
        "team_slots_truth": cov_teams,
        "teams_identified": seen_teams,
        "team_coverage": round(seen_teams / cov_teams, 4) if cov_teams else None,
        "player_name_coverage": round(len(with_player) / cov_players, 4) if cov_players else None,
    }

    # ---- zone circles ----
    truth_zone = {}
    for l in labels:
        if l.get("kind") == "zone":
            truth_zone[(round(float(l["t"]), 3), l["ring"])] = (
                float(l["cx_m"]), float(l["cy_m"]), float(l["r_m"]))
    zrows = [r for r in tracks if r.get("type") == "zone"]
    zstats: Dict[str, Any] = {}
    for ring in sorted({r["ring"] for r in zrows}):
        ce, re_, found, missing = [], [], 0, 0
        for r in [z for z in zrows if z["ring"] == ring]:
            if r.get("state") == "unknown":
                missing += 1
                continue
            key = (round(float(r["t"]), 3), ring)
            tv = truth_zone.get(key)
            if tv is None:
                continue
            found += 1
            ce.append(math.dist((float(r["cx_m"]), float(r["cy_m"])), (tv[0], tv[1])))
            re_.append(abs(float(r["r_m"]) - tv[2]))
        zstats[ring] = {"fitted": found, "unknown": missing,
                        "centre_error_m": _stats(ce), "radius_error_m": _stats(re_)}
    rep.zones = zstats

    # ---- behaviour layer: same derivation on truth and on extraction ----
    t_events = derive_events(truth_records(labels), prof)
    e_events = derive_events(tracks, prof)
    rep.behaviour = _compare_behaviour(t_events, e_events)
    return rep


def _team_landings(events: Sequence[Dict[str, Any]]) -> Dict[str, Tuple[float, float, float, str]]:
    """Earliest landing per team: (t, x, y, place)."""
    out: Dict[str, Tuple[float, float, float, str]] = {}
    for e in events:
        if e["type"] != "landing":
            continue
        cur = out.get(e["team"])
        if cur is None or float(e["t"]) < cur[0]:
            out[e["team"]] = (float(e["t"]), float(e["x_m"]), float(e["y_m"]), e["place"])
    return out


def _team_rotations(events: Sequence[Dict[str, Any]]) -> Dict[str, Tuple[int, float]]:
    out: Dict[str, Tuple[int, float]] = {}
    for e in events:
        if e["type"] != "rotation":
            continue
        n, dist = out.get(e["team"], (0, 0.0))
        out[e["team"]] = (n + 1, dist + float(e["distance_m"]))
    return out


def _compare_behaviour(truth: Sequence[Dict[str, Any]],
                       got: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare what the two logs say, at team level.

    Per-player comparison is not the right unit: four team-mates on a drop are
    one dot, so the stream cannot be asked who landed where individually. What
    it can be asked is where each team landed and how it moved, and that is
    what a coach reads anyway.
    """
    out: Dict[str, Any] = {}
    tp = [e for e in truth if e["type"] == "zone_phase"]
    gp = [e for e in got if e["type"] == "zone_phase"]
    pairs = list(zip(tp, gp))
    out["phases"] = {
        "truth": len(tp), "extracted": len(gp),
        "announce_time_error_s": round(statistics.median(
            [abs(float(a["t"]) - float(b["t"])) for a, b in pairs]), 2) if pairs else None,
        "centre_error_m": round(statistics.median(
            [math.dist((float(a["cx_m"]), float(a["cy_m"])),
                       (float(b["cx_m"]), float(b["cy_m"]))) for a, b in pairs]), 2)
        if pairs else None,
        "radius_error_m": round(statistics.median(
            [abs(float(a["r_m"]) - float(b["r_m"])) for a, b in pairs]), 2) if pairs else None,
    }

    tl, gl = _team_landings(truth), _team_landings(got)
    both = sorted(set(tl) & set(gl))
    place_ok = sum(1 for k in both if tl[k][3] == gl[k][3])
    derr = [math.dist((tl[k][1], tl[k][2]), (gl[k][1], gl[k][2])) for k in both]
    terr = [abs(tl[k][0] - gl[k][0]) for k in both]
    out["team_landings"] = {
        "truth_teams": len(tl), "extracted_teams": len(gl), "matched": len(both),
        "same_place": place_ok,
        "place_agreement": round(place_ok / len(both), 4) if both else None,
        "position_error_m": _stats(derr),
        "time_error_s": round(statistics.median(terr), 2) if terr else None,
    }

    tr, gr = _team_rotations(truth), _team_rotations(got)
    keys = sorted(set(tr) & set(gr))
    out["team_rotations"] = {
        "teams_with_rotations_truth": len(tr), "extracted": len(gr), "compared": len(keys),
        "median_count_difference": round(statistics.median(
            [gr[k][0] - tr[k][0] for k in keys]), 2) if keys else None,
        "median_distance_ratio": round(statistics.median(
            [gr[k][1] / tr[k][1] for k in keys if tr[k][1] > 0]), 3) if keys else None,
    }
    return out


def render_map_report(rep: MapReport, title: str = "Phase 3 map-stream accuracy") -> str:
    d = rep.as_dict()
    L = [f"# {title}", "",
         f"Position match tolerance ±{rep.tolerance_s:.2f}s; an identity counts as "
         f"correct when the dot is within {rep.identity_tol_m:.0f} m of a player it claims "
         f"to be.", "", "## Positions", ""]
    L.append("| measurement | n | median | mean | p90 | max | ≤25 m | ≤50 m | ≤100 m |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for name, st in (("named player vs truth", d["position"]),
                     ("dot vs nearest player of the team it claims", d["team_position"])):
        if st.get("n"):
            L.append(f"| {name} | {st['n']} | {st['median_m']} m | {st['mean_m']} m | "
                     f"{st['p90_m']} m | {st['max_m']} m | {st['within_25m']} | "
                     f"{st['within_50m']} | {st['within_100m']} |")
    i = d["identity"]
    L += ["", "## Identity", "",
          "| measurement | value |", "|---|---|",
          f"| samples emitted | {i['samples']} |",
          f"| samples with a team | {i['with_team']} |",
          f"| samples with a player name | {i['with_player']} |",
          f"| team identity precision | {i['team_precision']} |",
          f"| player identity precision | {i['player_precision']} |"]
    c = d["coverage"]
    L += ["", "## Coverage", "",
          "| measurement | value |", "|---|---|",
          f"| player-slots in truth | {c['player_slots_truth']} |",
          f"| covered by a dot (clusters counted by size) | {c['player_slots_covered_by_dots']} ({c['dot_coverage']}) |",
          f"| team-slots identified | {c['teams_identified']} of {c['team_slots_truth']} ({c['team_coverage']}) |",
          f"| samples carrying a player name | {c['player_name_coverage']} |"]
    L += ["", "## Zone circles", "",
          "| ring | fitted | unknown | centre error median | centre p90 | radius error median |",
          "|---|---|---|---|---|---|"]
    for ring, z in d["zones"].items():
        ce, re_ = z["centre_error_m"], z["radius_error_m"]
        L.append(f"| {ring} | {z['fitted']} | {z['unknown']} | "
                 f"{ce.get('median_m', '-')} m | {ce.get('p90_m', '-')} m | "
                 f"{re_.get('median_m', '-')} m |")
    b = d["behaviour"]
    pl = b["team_landings"]
    L += ["", "## Behaviour layer (same derivation run on truth and on extraction)", "",
          "| measurement | truth | extracted | agreement |", "|---|---|---|---|",
          f"| zone phases | {b['phases']['truth']} | {b['phases']['extracted']} | "
          f"announce ±{b['phases']['announce_time_error_s']} s, centre "
          f"{b['phases']['centre_error_m']} m, radius {b['phases']['radius_error_m']} m |",
          f"| team landings | {pl['truth_teams']} | {pl['extracted_teams']} | "
          f"{pl['matched']} matched, same place {pl['place_agreement']}, "
          f"median position error {pl['position_error_m'].get('median_m')} m, "
          f"median time error {pl['time_error_s']} s |",
          f"| team rotations | {b['team_rotations']['teams_with_rotations_truth']} teams | "
          f"{b['team_rotations']['extracted']} teams | "
          f"median count difference {b['team_rotations']['median_count_difference']}, "
          f"median distance ratio {b['team_rotations']['median_distance_ratio']} |"]
    if rep.notes:
        L += ["", "## Notes", ""] + [f"- {n}" for n in rep.notes]
    return "\n".join(L) + "\n"
