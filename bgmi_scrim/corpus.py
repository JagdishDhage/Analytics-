"""A corpus of matches, and what each team does across all of them.

One match at a time: `ingest` reads a map stream, derives what every team in
it did, and appends that to the corpus as facts. The video is never needed
again. `scout` then aggregates the corpus into one profile per team.

Two rules carried over from the rest of this toolchain. Nothing is estimated:
a figure that rests on too little evidence is reported as such, with the
evidence count beside it, rather than averaged into something that looks
solid. And every aggregate can name the matches it came from, so a coach can
go back to the clip.
"""
from __future__ import annotations

import json
import os
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .contacts import (contacts as derive_contacts, losses as derive_losses,
                       paths_from_samples, placements, team_eliminations)

#: corpus layout
MATCH_DIR = "matches"
CORPUS_VERSION = 1

#: below this many observations a per-team aggregate is reported as thin
MIN_MATCHES = 3
#: a zone-edge share needs this much time inside the circle to mean anything
EDGE_MIN_INSIDE_S = 30.0


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _median(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _share(n: int, d: int) -> Optional[float]:
    return (n / d) if d else None


# --------------------------------------------------------------------------
# one match
# --------------------------------------------------------------------------

def match_facts(match_id: str, tracks_path: str, events_path: str,
                played_at: Optional[str] = None) -> Dict[str, Any]:
    """Everything the corpus keeps about one match, for every team in it."""
    tracks = _read_jsonl(tracks_path)
    events = _read_jsonl(events_path)
    info = next((r for r in tracks if r.get("type") == "map_info"), {})
    samples = [r for r in tracks if r.get("type") == "sample"]
    if not samples:
        raise ValueError(f"{tracks_path} holds no position samples")
    end = max(float(r["t"]) for r in samples)

    paths = paths_from_samples(samples)
    losses = derive_losses(paths, end)
    elims = team_eliminations(paths, end)
    ranks = placements(elims)
    cts = derive_contacts(paths, end)

    phases = sorted((e for e in events if e.get("type") == "zone_phase"),
                    key=lambda e: e.get("phase", 0))
    teams = sorted({p.team for p in paths})
    per_team: Dict[str, Any] = {}
    for team in teams:
        te = [e for e in events if e.get("team") == team and e.get("unit") == "team"]
        landing = next((e for e in te if e.get("type") == "landing"), None)
        loot = [e for e in te if e.get("type") == "loot"]
        rots = [e for e in te if e.get("type") == "rotation"]
        zsum = next((e for e in te if e.get("type") == "zone_summary"), None)
        entries = [e for e in te if e.get("type") == "zone_entry"]
        spread = [float(e["value_m"]) for e in events
                  if e.get("type") == "squad_spread" and e.get("team") == team]
        my_losses = [l for l in losses if l.team == team]
        my_contacts = [c for c in cts if team in (c.team_a, c.team_b)]

        per_team[team] = {
            "team": team,
            "tracked_from_s": min((p.t0 for p in paths if p.team == team), default=None),
            "tracked_to_s": max((p.t1 for p in paths if p.team == team), default=None),
            "players_seen": len({p.player for p in paths if p.team == team and p.player}),
            "landing": None if landing is None else {
                "t": landing.get("t"), "place": landing.get("place"),
                "x_m": landing.get("x_m"), "y_m": landing.get("y_m"),
                "confidence": landing.get("confidence")},
            "loot_s": sum(float(e.get("duration_s") or 0.0) for e in loot),
            "rotations": [{
                "phase": r.get("phase"), "t": r.get("t"),
                "after_announce_s": r.get("after_announce_s"),
                "mode": r.get("mode"), "distance_m": r.get("distance_m"),
                "from": r.get("from"), "to": r.get("to"),
                "before_close_s": r.get("before_close_s"),
                "arrived_after_close_s": r.get("arrived_after_close_s"),
            } for r in rots],
            "zone": None if zsum is None else {
                "time_outside_blue_s": zsum.get("time_outside_blue_s"),
                "edge_time_fraction": zsum.get("edge_time_fraction"),
                "edge_time_inside_s": zsum.get("edge_time_inside_s"),
                "phases_entered": sum(1 for e in entries if e.get("state") != "never_entered"),
                "phases_never_entered": sum(1 for e in entries if e.get("state") == "never_entered"),
            },
            "squad_spread_median_m": _median(spread),
            "survival": {
                "eliminated_t": elims.get(team),
                "survived_to_end": elims.get(team) is None,
                "placement": ranks.get(team),
                "players_lost": len(my_losses),
                "first_loss_t": my_losses[0].t if my_losses else None,
            },
            "contacts": [c.as_dict() if c.team_a == team else _flip(c.as_dict())
                         for c in my_contacts],
        }

    return {
        "corpus_version": CORPUS_VERSION,
        "match_id": match_id,
        "played_at": played_at,
        "map": info.get("map"),
        "world_size_m": info.get("world_size_m"),
        "profile": info.get("profile"),
        "stream_length_s": round(end, 1),
        "teams_seen": len(teams),
        "zone_phases": len(phases),
        "contacts_total": len(cts),
        "sources": {"tracks": os.path.basename(tracks_path),
                    "map_events": os.path.basename(events_path)},
        "teams": per_team,
    }


def _flip(d: Dict[str, Any]) -> Dict[str, Any]:
    """The same contact seen from the other team's side."""
    out = dict(d)
    out["team"], out["opponent"] = d["opponent"], d["team"]
    out["losses_team"], out["losses_opponent"] = d["losses_opponent"], d["losses_team"]
    out["closed_team_m"], out["closed_opponent_m"] = d["closed_opponent_m"], d["closed_team_m"]
    out["mode_team"], out["mode_opponent"] = d["mode_opponent"], d["mode_team"]
    out["players_team"], out["players_opponent"] = d["players_opponent"], d["players_team"]
    return out


def write_match(corpus_dir: str, facts: Dict[str, Any], overwrite: bool = False) -> str:
    d = os.path.join(corpus_dir, MATCH_DIR)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{facts['match_id']}.json")
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f"{path} already in the corpus; pass --overwrite to replace it")
    with open(path, "w") as fh:
        json.dump(facts, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return path


def load_corpus(corpus_dir: str) -> List[Dict[str, Any]]:
    d = os.path.join(corpus_dir, MATCH_DIR)
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d)):
        if name.endswith(".json"):
            with open(os.path.join(d, name)) as fh:
                out.append(json.load(fh))
    return out


# --------------------------------------------------------------------------
# across matches
# --------------------------------------------------------------------------

@dataclass
class TeamProfile:
    team: str
    matches: List[str] = field(default_factory=list)
    maps: Counter = field(default_factory=Counter)

    drops: Counter = field(default_factory=Counter)
    drop_t: List[float] = field(default_factory=list)
    loot_s: List[float] = field(default_factory=list)

    rot_delays: List[float] = field(default_factory=list)
    rot_before_announce: int = 0
    rot_total: int = 0
    rot_vehicle: int = 0
    rot_distance_per_match: List[float] = field(default_factory=list)
    rot_late_arrivals: int = 0

    outside_blue: List[float] = field(default_factory=list)
    edge_fracs: List[float] = field(default_factory=list)
    edge_thin: int = 0
    never_entered: List[int] = field(default_factory=list)

    spread: List[float] = field(default_factory=list)

    placements: List[int] = field(default_factory=list)
    elim_t: List[float] = field(default_factory=list)
    survived: int = 0
    players_lost: List[int] = field(default_factory=list)

    contacts_per_match: List[int] = field(default_factory=list)
    contact_dists: List[float] = field(default_factory=list)
    contact_initiated: int = 0
    contact_received: int = 0
    contact_neutral: int = 0
    contact_vehicle: int = 0
    contact_total: int = 0
    losses_taken: List[int] = field(default_factory=list)
    losses_dealt: List[int] = field(default_factory=list)
    won: int = 0
    lost: int = 0
    even: int = 0
    bloodless: int = 0
    opponents: Counter = field(default_factory=Counter)

    def as_dict(self) -> Dict[str, Any]:
        n = len(self.matches)
        thin = n < MIN_MATCHES
        return {
            "team": self.team,
            "matches": n,
            "match_ids": self.matches,
            "thin_evidence": thin,
            "maps": dict(self.maps),
            "drop": {
                "places": [{"place": p, "matches": c, "share": round(c / n, 3)}
                           for p, c in self.drops.most_common(6)] if n else [],
                "median_landing_t_s": _median(self.drop_t),
                "median_loot_s_at_drop": _median(self.loot_s),
            },
            "rotation": {
                "rotations_per_match": round(self.rot_total / n, 2) if n else None,
                "median_delay_after_announce_s": _median(self.rot_delays),
                "share_moving_before_announce": _share(self.rot_before_announce, self.rot_total),
                "share_by_vehicle": _share(self.rot_vehicle, self.rot_total),
                "median_distance_per_match_m": _median(self.rot_distance_per_match),
                "arrivals_after_the_circle_closed": self.rot_late_arrivals,
            },
            "zone": {
                "median_time_outside_blue_s": _median(self.outside_blue),
                "median_edge_share": _median(self.edge_fracs),
                "edge_share_from_matches": len(self.edge_fracs),
                "matches_with_too_little_time_inside": self.edge_thin,
                "median_phases_never_entered": _median([float(x) for x in self.never_entered]),
            },
            "shape": {"median_squad_spread_m": _median(self.spread)},
            "survival": {
                "median_placement": _median([float(p) for p in self.placements]),
                "best_placement": min(self.placements) if self.placements else None,
                "median_elimination_t_s": _median(self.elim_t),
                "survived_the_stream": self.survived,
                "median_players_lost": _median([float(x) for x in self.players_lost]),
            },
            "fights": {
                "contacts_per_match": round(self.contact_total / n, 2) if n else None,
                "median_engagement_distance_m": _median(self.contact_dists),
                "share_initiated": _share(self.contact_initiated, self.contact_total),
                "share_received": _share(self.contact_received, self.contact_total),
                "share_undecided": _share(self.contact_neutral, self.contact_total),
                "share_arrived_by_vehicle": _share(self.contact_vehicle, self.contact_total),
                "record": {"won": self.won, "lost": self.lost, "even": self.even,
                           "no_losses_either_side": self.bloodless},
                "median_losses_taken_per_contact": _median([float(x) for x in self.losses_taken]),
                "median_losses_dealt_per_contact": _median([float(x) for x in self.losses_dealt]),
                "most_met": [{"opponent": o, "contacts": c}
                             for o, c in self.opponents.most_common(5)],
            },
        }


def team_profiles(matches: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate a corpus into one profile per team."""
    acc: Dict[str, TeamProfile] = {}
    for m in matches:
        mid = m.get("match_id", "?")
        for team, f in (m.get("teams") or {}).items():
            p = acc.get(team)
            if p is None:
                p = acc[team] = TeamProfile(team)
            p.matches.append(mid)
            if m.get("map"):
                p.maps[m["map"]] += 1

            land = f.get("landing") or {}
            if land.get("place"):
                p.drops[land["place"]] += 1
            if land.get("t") is not None:
                p.drop_t.append(float(land["t"]))
            if f.get("loot_s") is not None:
                p.loot_s.append(float(f["loot_s"]))

            dist = 0.0
            for r in f.get("rotations") or []:
                p.rot_total += 1
                d = r.get("after_announce_s")
                if d is None:
                    p.rot_before_announce += 1
                else:
                    p.rot_delays.append(float(d))
                if r.get("mode") == "vehicle":
                    p.rot_vehicle += 1
                dist += float(r.get("distance_m") or 0.0)
                if r.get("arrived_after_close_s") is not None:
                    p.rot_late_arrivals += 1
            p.rot_distance_per_match.append(dist)

            z = f.get("zone") or {}
            if z.get("time_outside_blue_s") is not None:
                p.outside_blue.append(float(z["time_outside_blue_s"]))
            ef, inside = z.get("edge_time_fraction"), z.get("edge_time_inside_s")
            if ef is not None and (inside or 0) >= EDGE_MIN_INSIDE_S:
                p.edge_fracs.append(float(ef))
            elif ef is not None:
                p.edge_thin += 1
            if z.get("phases_never_entered") is not None:
                p.never_entered.append(int(z["phases_never_entered"]))

            if f.get("squad_spread_median_m") is not None:
                p.spread.append(float(f["squad_spread_median_m"]))

            s = f.get("survival") or {}
            if s.get("placement") is not None:
                p.placements.append(int(s["placement"]))
            if s.get("eliminated_t") is not None:
                p.elim_t.append(float(s["eliminated_t"]))
            if s.get("survived_to_end"):
                p.survived += 1
            if s.get("players_lost") is not None:
                p.players_lost.append(int(s["players_lost"]))

            cs = f.get("contacts") or []
            p.contacts_per_match.append(len(cs))
            for c in cs:
                p.contact_total += 1
                if c.get("min_distance_m") is not None:
                    p.contact_dists.append(float(c["min_distance_m"]))
                init = c.get("initiator")
                if init is None:
                    p.contact_neutral += 1
                elif init == team:
                    p.contact_initiated += 1
                else:
                    p.contact_received += 1
                if c.get("mode_team") == "vehicle":
                    p.contact_vehicle += 1
                lt = int(c.get("losses_team") or 0)
                lo = int(c.get("losses_opponent") or 0)
                p.losses_taken.append(lt)
                p.losses_dealt.append(lo)
                if lt == 0 and lo == 0:
                    p.bloodless += 1
                elif lt == lo:
                    p.even += 1
                elif lo > lt:
                    p.won += 1
                else:
                    p.lost += 1
                if c.get("opponent"):
                    p.opponents[c["opponent"]] += 1

    return {t: p.as_dict() for t, p in sorted(acc.items())}
