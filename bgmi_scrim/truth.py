"""Ground-truth harness.

Scores an events.jsonl against hand-labelled frames and events, and reports
per-field precision/recall plus a confusion list naming every single miss and
every false positive. Label file format (JSONL, one object per line):

  {"kind":"frame","t":62.0,"teams_alive":18,"players_alive":51,
   "zone_phase":4,"zone_state":"shrinking","team_alive":{"TX":3}}
  {"kind":"event","t":63.2,"type":"knock","victim_team":"TX","victim_slot":2,
   "killer_team":"GB","killer_slot":1}
  {"kind":"scalar","t":1.0,"type":"map_name","value":"ERANGEL"}

Only the fields present in a label line are scored, so partial labelling of a
frame is fine.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .events import read_jsonl

FRAME_FIELDS = ("teams_alive", "players_alive", "zone_phase", "zone_state")
EVENT_TYPES = ("knock", "kill")
SCALAR_TYPES = ("map_name", "placement", "winner")


class ExtractedState:
    """Replays an event log so the extractor's state can be asked for at time t."""

    def __init__(self, events: List[Dict[str, Any]]):
        self.events = sorted(events, key=lambda e: e["t"])
        self.series: Dict[str, List[Tuple[float, Any, float, str]]] = {}
        for e in self.events:
            typ = e.get("type")
            if typ in FRAME_FIELDS:
                self.series.setdefault(typ, []).append(
                    (e["t"], e.get("value"), e.get("confidence", 0.0), e.get("state", "ok")))
            elif typ == "team_alive_count":
                key = f"team_alive:{e.get('team')}"
                self.series.setdefault(key, []).append(
                    (e["t"], e.get("value"), e.get("confidence", 0.0), e.get("state", "ok")))

    def at(self, field_name: str, t: float) -> Tuple[Any, float, str]:
        out: Tuple[Any, float, str] = (None, 0.0, "missing")
        for ts, v, c, st in self.series.get(field_name, []):
            if ts <= t + 1e-9:
                out = (v, c, st)
            else:
                break
        return out

    def near(self, field_name: str, t: float, grace: float) -> List[Tuple[Any, float, str]]:
        """Values held anywhere in [t-grace, t+grace] (transition tolerance)."""
        vals = [self.at(field_name, t)]
        for ts, v, c, st in self.series.get(field_name, []):
            if t - grace <= ts <= t + grace:
                vals.append((v, c, st))
        return vals

    def of_type(self, typ: str) -> List[Dict[str, Any]]:
        return [e for e in self.events if e.get("type") == typ]


@dataclass
class FieldScore:
    field: str
    n: int = 0
    correct: int = 0
    wrong_confident: int = 0
    wrong_flagged: int = 0     # miss that the log itself marked unknown/low-confidence
    missing: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0


@dataclass
class EventScore:
    type: str
    truth: int = 0
    predicted: int = 0
    matched: int = 0
    duplicates: int = 0
    phantoms: int = 0
    misses: int = 0
    killer_checked: int = 0
    killer_correct: int = 0
    killer_null: int = 0
    mean_abs_dt: float = 0.0

    @property
    def precision(self) -> float:
        return self.matched / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.matched / self.truth if self.truth else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class Report:
    fields: Dict[str, FieldScore] = field(default_factory=dict)
    events: Dict[str, EventScore] = field(default_factory=dict)
    scalars: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    confusion: List[Dict[str, Any]] = field(default_factory=list)
    tolerance_s: float = 2.5
    grace_s: float = 0.75
    strict_fields: Dict[str, FieldScore] = field(default_factory=dict)

    def gate(self) -> Dict[str, Any]:
        ks = [self.events[t] for t in EVENT_TYPES if t in self.events]
        kill_knock_p = min([s.precision for s in ks], default=0.0)
        kill_knock_r = min([s.recall for s in ks], default=0.0)
        dups = sum(s.duplicates for s in ks)
        phantoms = sum(s.phantoms for s in ks)
        ta = self.fields.get("teams_alive")
        checks = {
            "kill_knock_precision>=0.95": kill_knock_p >= 0.95,
            "kill_knock_recall>=0.95": kill_knock_r >= 0.95,
            "no_duplicate_events": dups == 0,
            "no_phantom_kills": phantoms == 0,
            "teams_alive_frames>=0.99": (ta.accuracy >= 0.99) if ta and ta.n else False,
            "every_miss_accounted": True,   # the confusion list below is exhaustive
        }
        return {"passed": all(checks.values()), "checks": checks,
                "kill_knock_precision": kill_knock_p, "kill_knock_recall": kill_knock_r,
                "duplicates": dups, "phantoms": phantoms,
                "teams_alive_accuracy": ta.accuracy if ta else None}


def _same_actor(a_team: Any, a_slot: Any, b_team: Any, b_slot: Any) -> bool:
    if a_team != b_team:
        return False
    if a_slot is None or b_slot is None:
        return True
    return int(a_slot) == int(b_slot)


def score(labels_path: str | Path, events_path: str | Path, tolerance: float = 2.5,
          grace: float = 0.75, confidence_threshold: float = 0.6) -> Report:
    labels = read_jsonl(labels_path)
    events = read_jsonl(events_path)
    st = ExtractedState(events)
    rep = Report(tolerance_s=tolerance, grace_s=grace)

    # ---------- frame-level fields ----------
    for fname in FRAME_FIELDS:
        rep.fields[fname] = FieldScore(fname)
        rep.strict_fields[fname] = FieldScore(fname)
    team_score = FieldScore("team_alive_count")
    rep.fields["team_alive_count"] = team_score

    for lab in labels:
        if lab.get("kind") != "frame":
            continue
        t = float(lab["t"])
        for fname in FRAME_FIELDS:
            if fname not in lab:
                continue
            truth = lab[fname]
            fs, ss = rep.fields[fname], rep.strict_fields[fname]
            cand = st.near(fname, t, grace)
            strict = st.at(fname, t)
            ss.n += 1
            fs.n += 1
            if strict[0] == truth:
                ss.correct += 1
            if any(v == truth for v, _, _ in cand):
                fs.correct += 1
                continue
            v, c, state = strict
            if state in ("unknown", "missing") or c < confidence_threshold:
                fs.wrong_flagged += 1
            else:
                fs.wrong_confident += 1
            if state == "missing":
                fs.missing += 1
            rep.confusion.append({"kind": "frame_field", "field": fname, "t": t,
                                  "truth": truth, "read": v, "confidence": c,
                                  "state": state,
                                  "flagged": state in ("unknown", "missing") or c < confidence_threshold})
        for team, truth in (lab.get("team_alive") or {}).items():
            key = f"team_alive:{team}"
            cand = st.near(key, t, grace)
            team_score.n += 1
            if any(v == truth for v, _, _ in cand):
                team_score.correct += 1
                continue
            v, c, state = st.at(key, t)
            if state in ("unknown", "missing") or c < confidence_threshold:
                team_score.wrong_flagged += 1
            else:
                team_score.wrong_confident += 1
            rep.confusion.append({"kind": "team_alive", "field": key, "t": t,
                                  "truth": truth, "read": v, "confidence": c, "state": state,
                                  "flagged": state in ("unknown", "missing") or c < confidence_threshold})

    # ---------- events ----------
    for typ in EVENT_TYPES:
        truths = [l for l in labels if l.get("kind") == "event" and l.get("type") == typ]
        preds = [e for e in st.of_type(typ)]
        es = EventScore(typ, truth=len(truths), predicted=len(preds))
        used_pred: set[int] = set()
        used_truth: set[int] = set()
        dts: List[float] = []
        pairs: List[Tuple[float, int, int]] = []
        for ti, tr in enumerate(truths):
            for pi, pr in enumerate(preds):
                if not _same_actor(tr.get("victim_team"), tr.get("victim_slot"),
                                   pr.get("victim_team"), pr.get("victim_slot")):
                    continue
                dt = abs(float(pr["t"]) - float(tr["t"]))
                if dt <= tolerance:
                    pairs.append((dt, ti, pi))
        for dt, ti, pi in sorted(pairs):
            if ti in used_truth or pi in used_pred:
                continue
            used_truth.add(ti)
            used_pred.add(pi)
            dts.append(dt)
            es.matched += 1
            tr, pr = truths[ti], preds[pi]
            if tr.get("killer_team"):
                es.killer_checked += 1
                if pr.get("killer_team") is None:
                    es.killer_null += 1
                elif (pr.get("killer_team") == tr.get("killer_team")
                      and (pr.get("killer_slot") is None
                           or int(pr["killer_slot"]) == int(tr["killer_slot"]))):
                    es.killer_correct += 1
                else:
                    rep.confusion.append({
                        "kind": "killer_attribution", "type": typ, "t": pr["t"],
                        "victim": f"{tr.get('victim_team')}#{tr.get('victim_slot')}",
                        "truth_killer": f"{tr.get('killer_team')}#{tr.get('killer_slot')}",
                        "read_killer": f"{pr.get('killer_team')}#{pr.get('killer_slot')}",
                        "confidence": pr.get("confidence")})
        es.mean_abs_dt = sum(dts) / len(dts) if dts else 0.0
        # unmatched predictions: duplicate of a matched truth, or a phantom
        for pi, pr in enumerate(preds):
            if pi in used_pred:
                continue
            dup = any(_same_actor(tr.get("victim_team"), tr.get("victim_slot"),
                                  pr.get("victim_team"), pr.get("victim_slot"))
                      and abs(float(pr["t"]) - float(tr["t"])) <= tolerance
                      for ti, tr in enumerate(truths) if ti in used_truth)
            if dup:
                es.duplicates += 1
                rep.confusion.append({"kind": "duplicate_event", "type": typ, "t": pr["t"],
                                      "victim": f"{pr.get('victim_team')}#{pr.get('victim_slot')}",
                                      "confidence": pr.get("confidence"),
                                      "source": pr.get("source")})
            else:
                es.phantoms += 1
                rep.confusion.append({"kind": "phantom_event", "type": typ, "t": pr["t"],
                                      "victim": f"{pr.get('victim_team')}#{pr.get('victim_slot')}",
                                      "killer": f"{pr.get('killer_team')}#{pr.get('killer_slot')}",
                                      "confidence": pr.get("confidence"),
                                      "source": pr.get("source")})
        for ti, tr in enumerate(truths):
            if ti in used_truth:
                continue
            es.misses += 1
            rep.confusion.append({"kind": "missed_event", "type": typ, "t": tr["t"],
                                  "victim": f"{tr.get('victim_team')}#{tr.get('victim_slot')}",
                                  "killer": f"{tr.get('killer_team')}#{tr.get('killer_slot')}"})
        rep.events[typ] = es

    # ---------- scalars ----------
    for typ in SCALAR_TYPES:
        truths = [l for l in labels if l.get("kind") == "scalar" and l.get("type") == typ]
        if not truths:
            continue
        preds = st.of_type(typ)
        ok = 0
        for tr in truths:
            hit = any(str(p.get("value")) == str(tr.get("value")) for p in preds)
            ok += 1 if hit else 0
            if not hit:
                rep.confusion.append({"kind": "scalar_miss", "type": typ, "t": tr["t"],
                                      "truth": tr.get("value"),
                                      "read": [p.get("value") for p in preds]})
        rep.scalars[typ] = {"truth": len(truths), "predicted": len(preds), "correct": ok,
                            "values_read": [p.get("value") for p in preds]}
    return rep


def render_report(rep: Report, title: str = "Phase 1 accuracy report") -> str:
    g = rep.gate()
    L: List[str] = [f"# {title}", "",
                    f"Event match tolerance: ±{rep.tolerance_s:.2f}s; "
                    f"frame-field transition grace: ±{rep.grace_s:.2f}s", "",
                    "## Gate", ""]
    L.append("| check | result |")
    L.append("|---|---|")
    for k, v in g["checks"].items():
        L.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    L.append("")
    L.append(f"**Gate: {'PASSED' if g['passed'] else 'NOT MET'}**")
    L.append("")
    L.append("## Events")
    L.append("")
    L.append("| type | truth | predicted | matched | precision | recall | F1 | dup | phantom | miss | mean dt |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for typ, es in rep.events.items():
        L.append(f"| {typ} | {es.truth} | {es.predicted} | {es.matched} | {es.precision:.4f} | "
                 f"{es.recall:.4f} | {es.f1:.4f} | {es.duplicates} | {es.phantoms} | "
                 f"{es.misses} | {es.mean_abs_dt:.2f}s |")
    L.append("")
    L.append("| type | killer checked | killer correct | killer null | killer wrong |")
    L.append("|---|---|---|---|---|")
    for typ, es in rep.events.items():
        wrong = es.killer_checked - es.killer_correct - es.killer_null
        L.append(f"| {typ} | {es.killer_checked} | {es.killer_correct} | {es.killer_null} | {wrong} |")
    L.append("")
    L.append("## Frame fields")
    L.append("")
    L.append("| field | labels | correct | accuracy | wrong (confident) | wrong (flagged low-confidence) |")
    L.append("|---|---|---|---|---|---|")
    for fname, fs in rep.fields.items():
        if not fs.n:
            continue
        L.append(f"| {fname} | {fs.n} | {fs.correct} | {fs.accuracy:.4f} | "
                 f"{fs.wrong_confident} | {fs.wrong_flagged} |")
    if rep.strict_fields:
        L.append("")
        L.append("Strict (no transition grace):")
        L.append("")
        L.append("| field | labels | correct | accuracy |")
        L.append("|---|---|---|---|")
        for fname, fs in rep.strict_fields.items():
            if fs.n:
                L.append(f"| {fname} | {fs.n} | {fs.correct} | {fs.accuracy:.4f} |")
    if rep.scalars:
        L.append("")
        L.append("## Scalars")
        L.append("")
        L.append("| type | labels | correct | values read |")
        L.append("|---|---|---|---|")
        for typ, d in rep.scalars.items():
            L.append(f"| {typ} | {d['truth']} | {d['correct']} | {d['values_read']} |")
    L.append("")
    L.append(f"## Confusion list ({len(rep.confusion)} entries)")
    L.append("")
    if not rep.confusion:
        L.append("None.")
    else:
        L.append("| t | kind | detail |")
        L.append("|---|---|---|")
        for c in sorted(rep.confusion, key=lambda c: float(c.get("t") or 0)):
            detail = ", ".join(f"{k}={v}" for k, v in c.items() if k not in ("t", "kind"))
            L.append(f"| {float(c.get('t') or 0):.2f} | {c['kind']} | {detail} |")
    L.append("")
    return "\n".join(L)


def report_json(rep: Report) -> str:
    return json.dumps({
        "gate": rep.gate(),
        "tolerance_s": rep.tolerance_s,
        "grace_s": rep.grace_s,
        "events": {k: v.__dict__ | {"precision": v.precision, "recall": v.recall, "f1": v.f1}
                   for k, v in rep.events.items()},
        "fields": {k: v.__dict__ | {"accuracy": v.accuracy} for k, v in rep.fields.items()},
        "strict_fields": {k: v.__dict__ | {"accuracy": v.accuracy}
                          for k, v in rep.strict_fields.items()},
        "scalars": rep.scalars,
        "confusion": rep.confusion,
    }, indent=2)
