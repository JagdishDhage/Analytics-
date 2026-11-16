"""Scouting reports: how a team plays, across every match in the corpus.

Same tone as the preparation report. Numbers, the evidence count behind them,
and no adjectives. A figure from too few matches says so rather than being
presented as a tendency.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .corpus import MIN_MATCHES


def _pct(v: Optional[float]) -> str:
    return "-" if v is None else f"{v * 100:.0f}%"


def _num(v: Optional[float], unit: str = "", dp: int = 0) -> str:
    if v is None:
        return "-"
    return f"{v:.{dp}f}{unit}"


def _mmss(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return f"{int(v) // 60:02d}:{int(v) % 60:02d}"


def corpus_summary(matches: Sequence[Dict[str, Any]],
                   profiles: Dict[str, Dict[str, Any]]) -> List[str]:
    L: List[str] = ["# Scouting corpus", ""]
    if not matches:
        L += ["No matches ingested yet. Run `ingest` on a map stream first.", ""]
        return L
    maps: Dict[str, int] = {}
    total_contacts = 0
    for m in matches:
        if m.get("map"):
            maps[m["map"]] = maps.get(m["map"], 0) + 1
        total_contacts += int(m.get("contacts_total") or 0)
    hours = sum(float(m.get("stream_length_s") or 0) for m in matches) / 3600.0
    L += [f"{len(matches)} matches, {len(profiles)} teams, {hours:.1f} hours of map stream. "
          f"{total_contacts} contacts derived.", ""]
    L += ["Maps: " + ", ".join(f"{k} {v}" for k, v in sorted(maps.items())), ""]
    thin = [p["team"] for p in profiles.values() if p["thin_evidence"]]
    if thin:
        L += [f"{len(thin)} teams appear in fewer than {MIN_MATCHES} matches; their "
              "figures are marked thin and should not be read as tendencies.", ""]
    L += ["## Teams by matches seen", "",
          "| team | matches | median placement | contacts/match | initiated | won | lost |",
          "|---|---|---|---|---|---|---|"]
    for p in sorted(profiles.values(), key=lambda q: (-q["matches"], q["team"])):
        f, s = p["fights"], p["survival"]
        L.append(f"| {p['team']}{' *' if p['thin_evidence'] else ''} | {p['matches']} | "
                 f"{_num(s['median_placement'], dp=1)} | "
                 f"{_num(f['contacts_per_match'], dp=1)} | {_pct(f['share_initiated'])} | "
                 f"{f['record']['won']} | {f['record']['lost']} |")
    L += ["", "`*` fewer than %d matches." % MIN_MATCHES, ""]
    return L


def team_section(p: Dict[str, Any]) -> List[str]:
    team, n = p["team"], p["matches"]
    L: List[str] = [f"## {team}", ""]
    note = (f"{n} matches" if n >= MIN_MATCHES
            else f"**{n} matches only** -- too few for any of this to be a tendency")
    L += [f"{note}. Matches: {', '.join(p['match_ids'][:12])}"
          + (" ..." if len(p["match_ids"]) > 12 else "") + ".", ""]

    d = p["drop"]
    L += ["### Drop and loot", ""]
    if d["places"]:
        L += ["| place | matches | share |", "|---|---|---|"]
        for row in d["places"]:
            L.append(f"| {row['place']} | {row['matches']} | {_pct(row['share'])} |")
        L.append("")
    else:
        L += ["No landing was read in any match.", ""]
    L += [f"Median landing at {_mmss(d['median_landing_t_s'])}, "
          f"{_num(d['median_loot_s_at_drop'], ' s')} looting where they landed.", ""]

    r = p["rotation"]
    L += ["### Rotation", "",
          "| measurement | value |", "|---|---|",
          f"| rotations per match | {_num(r['rotations_per_match'], dp=1)} |",
          f"| median delay after the circle is announced | {_num(r['median_delay_after_announce_s'], ' s')} |",
          f"| already moving when it was announced | {_pct(r['share_moving_before_announce'])} |",
          f"| by vehicle | {_pct(r['share_by_vehicle'])} |",
          f"| median distance moved per match | {_num(r['median_distance_per_match_m'], ' m')} |",
          f"| arrivals after the circle had closed | {r['arrivals_after_the_circle_closed']} |",
          ""]

    z = p["zone"]
    L += ["### Zone", "",
          "| measurement | value |", "|---|---|",
          f"| median time outside the blue | {_num(z['median_time_outside_blue_s'], ' s')} |",
          f"| median share of time inside spent on the edge | "
          f"{_pct(z['median_edge_share'])} (from {z['edge_share_from_matches']} matches) |",
          f"| matches with too little time inside to say | {z['matches_with_too_little_time_inside']} |",
          f"| median phases never entered | {_num(z['median_phases_never_entered'], dp=1)} |",
          f"| median squad spread | {_num(p['shape']['median_squad_spread_m'], ' m')} |",
          ""]

    f = p["fights"]
    rec = f["record"]
    L += ["### Fights", "",
          "A contact is two teams within 120 m of each other. With no kill feed "
          "in a map stream, a loss is a dot that stopped appearing, and the "
          "initiator is whichever side closed the ground -- not who shot first.",
          "",
          "| measurement | value |", "|---|---|",
          f"| contacts per match | {_num(f['contacts_per_match'], dp=1)} |",
          f"| median engagement distance | {_num(f['median_engagement_distance_m'], ' m')} |",
          f"| they closed the ground | {_pct(f['share_initiated'])} |",
          f"| the opponent closed it | {_pct(f['share_received'])} |",
          f"| neither clearly did | {_pct(f['share_undecided'])} |",
          f"| arrived by vehicle | {_pct(f['share_arrived_by_vehicle'])} |",
          f"| record (won / lost / even / no losses) | "
          f"{rec['won']} / {rec['lost']} / {rec['even']} / {rec['no_losses_either_side']} |",
          f"| median losses taken per contact | {_num(f['median_losses_taken_per_contact'], dp=1)} |",
          f"| median losses dealt per contact | {_num(f['median_losses_dealt_per_contact'], dp=1)} |",
          ""]
    if f["most_met"]:
        L += ["Most met: " + ", ".join(f"{o['opponent']} ({o['contacts']})"
                                       for o in f["most_met"]), ""]

    s = p["survival"]
    L += ["### Survival", "",
          "| measurement | value |", "|---|---|",
          f"| median placement | {_num(s['median_placement'], dp=1)} |",
          f"| best placement | {s['best_placement'] if s['best_placement'] is not None else '-'} |",
          f"| median elimination time | {_mmss(s['median_elimination_t_s'])} |",
          f"| still alive when the stream ended | {s['survived_the_stream']} of {n} |",
          f"| median players lost per match | {_num(s['median_players_lost'], dp=1)} |",
          ""]
    return L


def not_measurable() -> List[str]:
    return [
        "## Not measurable from a map stream", "",
        "* Who killed whom, and damage. A map stream with no kill feed never "
        "says it. Losses here are dots that stopped appearing, and a contact "
        "reports what each side lost, not who did it.",
        "* Knocks, revives and whether a knocked player was traded. A dot does "
        "not change when its player goes down.",
        "* Weapons, utility, loot quality and inventory.",
        "* Elevation, buildings, and whether a team holds a compound or its "
        "open ground: the map is a flat top-down view.",
        "* Which player of a squad is which while their dots overlap. At map "
        "scale team-mates standing together are one dot.",
        "* Placement points, unless the organiser's table is supplied "
        "separately: placement here is the order teams stopped appearing.",
        "* Anything in a match whose stream was cut before the end. A team "
        "still on the map when the stream stops is recorded as surviving it, "
        "which is not the same as placing.",
        "",
    ]


def scout_markdown(matches: Sequence[Dict[str, Any]],
                   profiles: Dict[str, Dict[str, Any]],
                   teams: Optional[Sequence[str]] = None) -> str:
    L = corpus_summary(matches, profiles)
    want = list(teams) if teams else sorted(profiles)
    missing = [t for t in want if t not in profiles]
    for t in want:
        if t in profiles:
            L += team_section(profiles[t])
    if missing:
        L += [f"No corpus entry for: {', '.join(missing)}.", ""]
    L += not_measurable()
    return "\n".join(L)
