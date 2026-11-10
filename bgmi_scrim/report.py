"""Phase 2: team preparation report, built only from phase-1 event logs.

Nothing is inferred that the event log does not contain. A metric that needs
data phase 1 does not extract (positions, damage, utility, rotations) is not
estimated; it is listed under "Not measurable yet". Every figure carries the
match number and the video timestamp it came from.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .events import read_jsonl

# Placement points are an organiser rule, not something the footage contains,
# so the table is data with a documented default (BGMI/PMGO squad scheme).
DEFAULT_PLACEMENT_POINTS = {1: 10, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 8: 1}
KILL_POINTS = 1
TRADE_WINDOW_S = 15.0
CONVERSION_WINDOW_S = 60.0

NOT_MEASURABLE = [
    "Positioning, rotations, zone-edge play, vehicle time: needs minimap or "
    "positional data; phase 1 reads no map pixels.",
    "Damage dealt or taken, weapons, attachments, utility (grenades, smokes): "
    "not shown in the spectator HUD regions phase 1 reads.",
    "Who revived whom, revive timings and self-revive: a revive shows only as a "
    "slot going back to alive; phase 1 does not emit revive events.",
    "Third-party detection, fight-engagement counts, holding versus pushing: "
    "needs positions of both teams.",
    "Loot quality, landing spot, drop congestion: not in the event log.",
    "Individual player names: the panel is read as team tag plus slot number, so "
    "per-player identity across matches is slot-based only.",
]


def _fmt_ts(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


@dataclass
class Match:
    number: int
    source: str
    match_index: Optional[int]
    events: List[Dict[str, Any]]
    start_t: float = 0.0
    end_t: float = 0.0

    def of(self, *types: str) -> List[Dict[str, Any]]:
        return [e for e in self.events if e.get("type") in types]

    def cite(self, t: float) -> str:
        return f"M{self.number} {_fmt_ts(t)}"


def _split_matches(paths: Sequence[str], threshold: float) -> List[Match]:
    matches: List[Match] = []
    n = 0
    for p in paths:
        evs = [e for e in read_jsonl(p)
               if e.get("state", "ok") == "ok" and float(e.get("confidence", 0)) >= threshold]
        idxs = sorted({e.get("match_index", 0) or 0 for e in evs})
        for mi in idxs:
            sub = [e for e in evs if (e.get("match_index", 0) or 0) == mi]
            if not sub:
                continue
            n += 1
            starts = [e["t"] for e in sub if e.get("type") == "match_start"]
            ends = [e["t"] for e in sub if e.get("type") == "match_end"]
            m = Match(n, Path(p).name, mi, sorted(sub, key=lambda e: e["t"]),
                      start_t=min(starts) if starts else min(e["t"] for e in sub),
                      end_t=max(ends) if ends else max(e["t"] for e in sub))
            matches.append(m)
    return matches


def _dropped_counts(paths: Sequence[str], threshold: float) -> Dict[str, int]:
    low = unknown = total = 0
    for p in paths:
        for e in read_jsonl(p):
            total += 1
            if e.get("state") == "unknown":
                unknown += 1
            elif float(e.get("confidence", 0)) < threshold:
                low += 1
    return {"total": total, "unknown": unknown, "low_confidence": low}


@dataclass
class MatchFacts:
    m: Match
    team: str
    placement: Optional[int] = None
    placement_cite: str = ""
    winner: Optional[str] = None
    map_name: Optional[str] = None
    our_kills: List[Dict[str, Any]] = field(default_factory=list)
    our_knocks: List[Dict[str, Any]] = field(default_factory=list)
    our_deaths: List[Dict[str, Any]] = field(default_factory=list)
    our_knocked: List[Dict[str, Any]] = field(default_factory=list)
    full_wipe: bool = False
    alive_times: Dict[int, Tuple[float, str]] = field(default_factory=dict)

    @property
    def kill_points(self) -> int:
        return len(self.our_kills) * KILL_POINTS


def _phase_at(m: Match, t: float) -> Optional[int]:
    val = None
    for e in m.of("zone_phase"):
        if e["t"] <= t:
            val = e.get("value")
        else:
            break
    return val


def _collect(m: Match, team: str) -> MatchFacts:
    f = MatchFacts(m, team)
    for e in m.of("placement"):
        if e.get("team") in (None, team) and e.get("value") is not None:
            f.placement = int(e["value"])
            f.placement_cite = m.cite(e["t"])
    for e in m.of("winner"):
        f.winner = e.get("value")
    for e in m.of("map_name"):
        f.map_name = e.get("value")
    for e in m.of("kill"):
        if e.get("killer_team") == team:
            f.our_kills.append(e)
        if e.get("victim_team") == team:
            f.our_deaths.append(e)
    for e in m.of("knock"):
        if e.get("killer_team") == team:
            f.our_knocks.append(e)
        if e.get("victim_team") == team:
            f.our_knocked.append(e)
    # time alive per slot: a slot with no kill event against it survived to match end
    slots = sorted({int(e["victim_slot"]) for e in f.our_deaths if e.get("victim_slot")})
    for s in range(1, 5):
        deaths = [e for e in f.our_deaths if int(e.get("victim_slot") or 0) == s]
        if deaths:
            d = min(deaths, key=lambda e: e["t"])
            f.alive_times[s] = (float(d["t"]) - m.start_t, m.cite(d["t"]))
        else:
            f.alive_times[s] = (m.end_t - m.start_t, m.cite(m.end_t) + " (alive at log end)")
    f.full_wipe = len({int(e.get("victim_slot") or 0) for e in f.our_deaths} - {0}) >= 4
    return f


def build_report(event_paths: Sequence[str], team: str, block_name: str = "scrim block",
                 placement_points: Optional[Dict[int, int]] = None,
                 confidence_threshold: float = 0.6,
                 asked_for: Optional[Sequence[str]] = None) -> str:
    pts = {int(k): int(v) for k, v in (placement_points or DEFAULT_PLACEMENT_POINTS).items()}
    matches = _split_matches(event_paths, confidence_threshold)
    facts = [_collect(m, team) for m in matches]
    quality = _dropped_counts(event_paths, confidence_threshold)
    L: List[str] = []
    L.append(f"# {team} — {block_name} preparation report")
    L.append("")
    L.append(f"Matches in block: {len(matches)} "
             f"({'; '.join(f'M{m.number}={m.source}' for m in matches) or 'none'}). "
             f"Events read at confidence >= {confidence_threshold:.2f}; "
             f"{quality['unknown']} unknown-state and {quality['low_confidence']} "
             f"low-confidence records of {quality['total']} were excluded.")
    L.append("")
    L.append(f"Placement points table used (organiser rule, supplied as data, not read from "
             f"footage): {json.dumps({str(k): v for k, v in sorted(pts.items())})}; "
             f"{KILL_POINTS} point per kill. Ranks outside the table score 0.")
    L.append("")
    if not matches:
        L.append("No matches in the supplied logs.")
        return "\n".join(L)

    # ---------------- 1. Result ----------------
    L.append("## 1. Result")
    L.append("")
    L.append("| match | map | placement | placement pts | kills | kill pts | total | cumulative |")
    L.append("|---|---|---|---|---|---|---|---|")
    cum = 0
    totals: List[int] = []
    for f in facts:
        pp = pts.get(f.placement, 0) if f.placement is not None else 0
        tot = pp + f.kill_points
        cum += tot
        totals.append(tot)
        place = (f"#{f.placement} ({f.placement_cite})" if f.placement is not None
                 else "not in log")
        pp_s = str(pp) if f.placement is not None else "n/a"
        L.append(f"| M{f.m.number} | {f.map_name or 'not in log'} | {place} | {pp_s} | "
                 f"{len(f.our_kills)} | {f.kill_points} | {tot} | {cum} |")
    avg = sum(totals) / len(totals)
    L.append("")
    L.append(f"Block average total per match: {avg:.2f}. "
             + "; ".join(f"M{f.m.number} {('+' if totals[i]-avg>=0 else '')}{totals[i]-avg:.2f}"
                         for i, f in enumerate(facts)) + ".")
    missing_place = [f"M{f.m.number}" for f in facts if f.placement is None]
    if missing_place:
        L.append("")
        L.append(f"Placement absent from the log for {', '.join(missing_place)}; "
                 f"no placement points are claimed for those matches.")
    L.append("")

    # ---------------- 2. Survival ----------------
    L.append("## 2. Survival")
    L.append("")
    L.append("| match | avg time alive per player | first death | last death | deaths before zone 3 | full wipe | top 5 |")
    L.append("|---|---|---|---|---|---|---|")
    pre3_all: List[Tuple[int, Dict[str, Any]]] = []
    for f in facts:
        secs = [v[0] for v in f.alive_times.values()]
        avg_alive = sum(secs) / len(secs) if secs else 0.0
        deaths = sorted(f.our_deaths, key=lambda e: e["t"])
        first = f.m.cite(deaths[0]["t"]) if deaths else "no death in log"
        last = f.m.cite(deaths[-1]["t"]) if deaths else "no death in log"
        pre3 = [e for e in deaths
                if (_phase_at(f.m, e["t"]) or 0) and int(_phase_at(f.m, e["t"]) or 0) < 3]
        pre3_all.extend((f.m.number, e) for e in pre3)
        top5 = ("yes" if f.placement is not None and f.placement <= 5
                else ("no" if f.placement is not None else "placement not in log"))
        L.append(f"| M{f.m.number} | {avg_alive/60:.2f} min | {first} | {last} | "
                 f"{len(pre3)} | {'yes' if f.full_wipe else 'no'} | {top5} |")
    L.append("")
    if pre3_all:
        L.append("Deaths before zone 3: " + "; ".join(
            f"M{n} {_fmt_ts(e['t'])} slot {e.get('victim_slot')} "
            f"(killer {e.get('killer_team') or 'unattributed'})" for n, e in pre3_all) + ".")
    else:
        L.append("No deaths before zone 3 in the block.")
    L.append("")
    L.append("Per-player time alive, by slot (slot numbers are panel positions, "
             "not named players):")
    L.append("")
    L.append("| match | slot 1 | slot 2 | slot 3 | slot 4 |")
    L.append("|---|---|---|---|---|")
    for f in facts:
        cells = []
        for s in range(1, 5):
            secs, cite = f.alive_times[s]
            cells.append(f"{secs/60:.2f} min ({cite})")
        L.append(f"| M{f.m.number} | " + " | ".join(cells) + " |")
    L.append("")

    # ---------------- 3. Fights ----------------
    L.append("## 3. Fights")
    L.append("")
    L.append("| match | phase | our kills | kill timestamps | our deaths | death timestamps |")
    L.append("|---|---|---|---|---|---|")
    for f in facts:
        by_phase: Dict[Any, Dict[str, List[float]]] = {}
        for e in f.our_kills:
            by_phase.setdefault(_phase_at(f.m, e["t"]), {"k": [], "d": []})["k"].append(e["t"])
        for e in f.our_deaths:
            by_phase.setdefault(_phase_at(f.m, e["t"]), {"k": [], "d": []})["d"].append(e["t"])
        for ph in sorted(by_phase, key=lambda p: (p is None, p)):
            kt = by_phase[ph]["k"]
            dt = by_phase[ph]["d"]
            L.append(f"| M{f.m.number} | {ph if ph is not None else 'not in log'} | {len(kt)} | "
                     + (", ".join(f.m.cite(x) for x in kt) or "-") + f" | {len(dt)} | "
                     + (", ".join(f.m.cite(x) for x in dt) or "-") + " |")
    L.append("")

    # trade rate
    trades = 0
    trade_chances = 0
    trade_examples: List[str] = []
    for f in facts:
        ours_hit = sorted(f.our_knocked + [e for e in f.our_deaths
                                           if e not in f.our_knocked], key=lambda e: e["t"])
        for e in f.our_knocked:
            killer = e.get("killer_team")
            if not killer:
                continue
            trade_chances += 1
            avenged = [x for x in (f.our_knocks + f.our_kills)
                       if x.get("victim_team") == killer
                       and 0 <= float(x["t"]) - float(e["t"]) <= TRADE_WINDOW_S]
            if avenged:
                trades += 1
                if len(trade_examples) < 3:
                    trade_examples.append(
                        f"{f.m.cite(e['t'])} slot {e.get('victim_slot')} knocked by {killer}, "
                        f"traded {float(avenged[0]['t'])-float(e['t']):.1f}s later "
                        f"({f.m.cite(avenged[0]['t'])})")
    L.append(f"Trade rate (a teammate knocked or killed someone from the same opponent team "
             f"within {TRADE_WINDOW_S:.0f}s of our player being knocked): "
             f"{trades}/{trade_chances}"
             + (f" = {trades/trade_chances:.2f}" if trade_chances else " (no attributed knocks against us)")
             + ".")
    if trade_examples:
        L.append("")
        L.append("Examples: " + "; ".join(trade_examples) + ".")
    unattributed_knocks = sum(1 for f in facts for e in f.our_knocked if not e.get("killer_team"))
    if unattributed_knocks:
        L.append("")
        L.append(f"{unattributed_knocks} knocks against us have no killer in the log "
                 f"(kill feed unreadable at that moment) and are excluded from the trade "
                 f"denominator.")
    L.append("")

    # conversions
    conv = conv_den = 0
    conv_miss: List[str] = []
    for f in facts:
        for e in f.our_knocks:
            conv_den += 1
            vt, vs = e.get("victim_team"), e.get("victim_slot")
            finished = [x for x in f.m.of("kill")
                        if x.get("victim_team") == vt and x.get("victim_slot") == vs
                        and 0 <= float(x["t"]) - float(e["t"]) <= CONVERSION_WINDOW_S]
            if finished:
                conv += 1
            elif len(conv_miss) < 5:
                conv_miss.append(f"{f.m.cite(e['t'])} {vt}#{vs}")
    L.append(f"Knocks we got converted to kills within {CONVERSION_WINDOW_S:.0f}s: "
             f"{conv}/{conv_den}"
             + (f" = {conv/conv_den:.2f}" if conv_den else "") + ".")
    if conv_miss:
        L.append("")
        L.append("Unconverted knocks (revived or not finished in the log): "
                 + "; ".join(conv_miss) + ".")
    # knocks conceded then lost
    lost = lost_den = 0
    lost_examples: List[str] = []
    for f in facts:
        for e in f.our_knocked:
            lost_den += 1
            vs = e.get("victim_slot")
            died = [x for x in f.our_deaths
                    if x.get("victim_slot") == vs
                    and 0 <= float(x["t"]) - float(e["t"]) <= CONVERSION_WINDOW_S]
            if died:
                lost += 1
                if len(lost_examples) < 3:
                    lost_examples.append(
                        f"{f.m.cite(e['t'])} slot {vs} knocked, dead "
                        f"{float(died[0]['t'])-float(e['t']):.1f}s later")
    L.append("")
    L.append(f"Knocks conceded and then lost (our knocked player died within "
             f"{CONVERSION_WINDOW_S:.0f}s): {lost}/{lost_den}"
             + (f" = {lost/lost_den:.2f}" if lost_den else "") + ".")
    if lost_examples:
        L.append("")
        L.append("Examples: " + "; ".join(lost_examples) + ".")
    L.append("")

    # ---------------- 4. Opponents ----------------
    L.append("## 4. Opponents")
    L.append("")
    taken: Dict[str, List[Tuple[int, float, Any]]] = {}
    unattributed = 0
    for f in facts:
        for e in f.our_deaths:
            kt = e.get("killer_team")
            if not kt:
                unattributed += 1
                continue
            taken.setdefault(kt, []).append((f.m.number, float(e["t"]),
                                             _phase_at(f.m, e["t"])))
    if taken:
        L.append("| opponent | our lives taken | phases | timestamps |")
        L.append("|---|---|---|---|")
        for opp, rows in sorted(taken.items(), key=lambda kv: -len(kv[1])):
            phases = ", ".join(str(r[2] if r[2] is not None else "?") for r in rows)
            stamps = ", ".join(f"M{r[0]} {_fmt_ts(r[1])}" for r in rows)
            L.append(f"| {opp} | {len(rows)} | {phases} | {stamps} |")
    else:
        L.append("No deaths against us carry a killer team in the log.")
    L.append("")
    L.append(f"Deaths with no killer in the log: {unattributed}"
             f"{' of ' + str(sum(len(f.our_deaths) for f in facts)) if facts else ''}. "
             f"Those are excluded from this table rather than assigned.")
    L.append("")

    # ---------------- 5. Practice focuses ----------------
    L.append("## 5. Practice focuses")
    L.append("")
    focuses = _focuses(facts, trades, trade_chances, conv, conv_den, lost, lost_den, pre3_all)
    for i, (title, evidence, target) in enumerate(focuses[:3], start=1):
        L.append(f"{i}. **{title}**")
        L.append(f"   - Evidence: {evidence}")
        L.append(f"   - Target next block: {target}")
    if not focuses:
        L.append("The log holds too few attributed events in this block to rank focuses.")
    L.append("")

    # ---------------- 6. Not measurable yet ----------------
    L.append("## 6. Not measurable yet")
    L.append("")
    for item in NOT_MEASURABLE:
        L.append(f"- {item}")
    for ask in (asked_for or []):
        L.append(f"- Asked for: \"{ask}\" — the phase-1 event log has no field that supports it.")
    L.append("")
    return "\n".join(L)


def _focuses(facts: List[MatchFacts], trades: int, trade_chances: int, conv: int,
             conv_den: int, lost: int, lost_den: int,
             pre3: List[Tuple[int, Dict[str, Any]]]) -> List[Tuple[str, str, str]]:
    """Rank candidate focuses by the size of the measured gap."""
    out: List[Tuple[float, Tuple[str, str, str]]] = []
    if trade_chances:
        rate = trades / trade_chances
        worst = None
        for f in facts:
            for e in f.our_knocked:
                if not e.get("killer_team"):
                    continue
                avenged = [x for x in (f.our_knocks + f.our_kills)
                           if x.get("victim_team") == e.get("killer_team")
                           and 0 <= float(x["t"]) - float(e["t"]) <= TRADE_WINDOW_S]
                if not avenged:
                    worst = (f.m.cite(e["t"]), e)
                    break
            if worst:
                break
        ev = (f"trade rate {trades}/{trade_chances} = {rate:.2f}; "
              + (f"first untraded knock {worst[0]}, slot {worst[1].get('victim_slot')} "
                 f"down to {worst[1].get('killer_team')} with no answer inside "
                 f"{TRADE_WINDOW_S:.0f}s" if worst else "no untraded knock in the block"))
        out.append((1.0 - rate, ("Trade discipline on first contact", ev,
                                 f"trade rate >= {min(0.95, rate + 0.15):.2f} next block")))
    if conv_den:
        rate = conv / conv_den
        out.append((1.0 - rate, (
            "Finishing knocks",
            f"{conv}/{conv_den} = {rate:.2f} of our knocks ended in a kill inside "
            f"{CONVERSION_WINDOW_S:.0f}s",
            f"conversion >= {min(0.95, rate + 0.10):.2f} next block")))
    if lost_den:
        rate = lost / lost_den
        out.append((rate, (
            "Saving knocked teammates",
            f"{lost}/{lost_den} = {rate:.2f} of knocks conceded turned into a death inside "
            f"{CONVERSION_WINDOW_S:.0f}s",
            f"keep that below {max(0.05, rate - 0.10):.2f} next block")))
    if pre3:
        n = len(pre3)
        per_match = n / max(1, len({m for m, _ in pre3}))
        first = pre3[0]
        out.append((min(1.0, n / 8.0), (
            "Early-phase survival",
            f"{n} deaths before zone 3 in the block; first at M{first[0]} "
            f"{_fmt_ts(first[1]['t'])} slot {first[1].get('victim_slot')} to "
            f"{first[1].get('killer_team') or 'unattributed'}",
            f"at most {max(0, int(per_match) - 1)} deaths before zone 3 per match")))
    wipes = [f for f in facts if f.full_wipe]
    if wipes:
        rate = len(wipes) / len(facts)
        last = sorted(wipes[0].our_deaths, key=lambda e: e["t"])[-1] if wipes[0].our_deaths else None
        out.append((rate, (
            "Avoiding full wipes",
            f"{len(wipes)} of {len(facts)} matches ended in a full wipe"
            + (f"; earliest example M{wipes[0].m.number} {_fmt_ts(last['t'])}" if last else ""),
            f"at most {max(0, len(wipes) - 1)} full wipes next block")))
    out.sort(key=lambda x: -x[0])
    return [o[1] for o in out]
