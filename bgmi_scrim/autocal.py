"""Propose a profile from the footage itself.

Calibrating by hand is the slowest part of putting this on a new organiser's
overlay, and it is the step that stops a VOD being analysed at all. Nothing
here is trusted on its own: every proposal is a starting point that
`calibrate --preview` draws back onto a frame so a person can see whether the
boxes sit where the HUD is. Each proposal carries the evidence it was made
from, so a wrong one is obvious rather than silent.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .ocr import ALNUM, DIGITS, read_text, tesseract_available
from .preprocess import prepare_roi, roi_is_blank


@dataclass
class Proposal:
    """A profile fragment, with what it was inferred from."""

    fields: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def note(self, text: str) -> None:
        self.notes.append(text)


def _cluster(values: Sequence[float], tol: float) -> List[List[float]]:
    """One-dimensional single-link clustering."""
    out: List[List[float]] = []
    for v in sorted(values):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return out


def _median_gap(centres: Sequence[float]) -> Optional[float]:
    xs = sorted(centres)
    gaps = [b - a for a, b in zip(xs, xs[1:])]
    return float(np.median(gaps)) if gaps else None


def _saturated_blobs(frame: np.ndarray, sat_min: int = 80, val_min: int = 70,
                     area: Tuple[int, int] = (20, 3000),
                     min_circularity: float = 0.0
                     ) -> List[Tuple[int, int, int, int, int]]:
    """(x, y, w, h, area) of small saturated patches: slot pips and map dots."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, sat_min, val_min], np.uint8),
                       np.array([179, 255, 255], np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if not (area[0] <= a <= area[1]):
            continue
        if h == 0 or not (0.45 <= w / float(h) <= 2.2):
            continue
        if min_circularity > 0:
            peri = 2.0 * (w + h)
            circ = 4.0 * math.pi * float(a) / (peri * peri) if peri else 0.0
            if circ < min_circularity:
                continue        # a piece of the zone ring, not a player
        out.append((int(x), int(y), int(w), int(h), int(a)))
    return out


# ---------------- the spectator HUD ----------------
def propose_team_panel(frames: Sequence[np.ndarray]) -> Proposal:
    """Find the team list: many evenly spaced rows of small coloured pips."""
    p = Proposal()
    best: Optional[Dict[str, Any]] = None
    for frame in frames:
        blobs = _saturated_blobs(frame)
        if len(blobs) < 12:
            continue
        hs = [b[3] for b in blobs]
        tol = max(3.0, float(np.median(hs)) * 0.6)
        rows = _cluster([b[1] + b[3] / 2 for b in blobs], tol)
        rows = [r for r in rows if len(r) >= 2]
        if len(rows) < 6:
            continue
        row_centres = [float(np.mean(r)) for r in rows]
        pitch = _median_gap(row_centres)
        if not pitch or pitch <= 0:
            continue
        # keep the rows that sit on that pitch: a panel is a regular stack
        keep = [c for c in row_centres
                if any(abs((c - o) % pitch) < 0.25 * pitch or
                       abs((c - o) % pitch - pitch) < 0.25 * pitch
                       for o in row_centres if o != c)]
        if len(keep) < 6:
            continue
        in_rows = [b for b in blobs
                   if any(abs(b[1] + b[3] / 2 - c) <= tol for c in keep)]
        cols = _cluster([b[0] + b[2] / 2 for b in in_rows],
                        max(3.0, float(np.median([b[2] for b in in_rows])) * 0.6))
        cols = [c for c in cols if len(c) >= max(3, len(keep) // 3)]
        if len(cols) < 2:
            continue
        col_centres = sorted(float(np.mean(c)) for c in cols)
        slot_pitch = _median_gap(col_centres) or 0.0
        slot_w = int(round(float(np.median([b[2] for b in in_rows]))))
        slot_h = int(round(float(np.median([b[3] for b in in_rows]))))
        top = min(b[1] for b in in_rows)
        left = int(round(col_centres[0] - slot_w / 2))
        score = len(keep) * len(cols)
        cand = {"rows": len(keep), "slots": len(cols), "row_pitch": round(pitch, 2),
                "slot_pitch": round(slot_pitch, 2), "slot_w": slot_w, "slot_h": slot_h,
                "top": int(top), "left": left, "score": score}
        if best is None or score > best["score"]:
            best = cand
    if best is None:
        p.note("no regular stack of coloured pips found: the team list may be "
               "styled differently, or the frames sampled were pre-match")
        return p

    tag_w, tag_left = _measure_tag_column(frames, best)
    panel_x = max(0, tag_left - 4)
    panel_y = max(0, best["top"] - 4)
    p.fields = {
        "team_panel": {
            "box": [panel_x, panel_y,
                    int(best["left"] - panel_x + best["slot_pitch"] * best["slots"] + 8),
                    int(best["row_pitch"] * best["rows"] + best["slot_h"])],
            "kind": "team_panel",
            "rows": int(best["rows"]), "slots": int(best["slots"]),
            "row_pitch": best["row_pitch"],
            "tag_x": 0, "tag_dy": 0, "tag_w": tag_w,
            "tag_h": int(best["row_pitch"] * 0.78), "tag_min_score": 0.6,
            "slot_x0": int(best["left"] - panel_x), "slot_pitch": best["slot_pitch"],
            "slot_dy": int(best["top"] - panel_y), "slot_w": best["slot_w"],
            "slot_h": best["slot_h"],
        }
    }
    p.confidence = min(0.9, 0.3 + 0.03 * best["rows"] + 0.1 * best["slots"])
    p.note(f"{best['rows']} rows of {best['slots']} pips on a "
           f"{best['row_pitch']:.1f}px pitch")
    return p


def _measure_tag_column(frames: Sequence[np.ndarray],
                        panel: Dict[str, Any]) -> Tuple[int, int]:
    """How far left the team tags run, measured from the text beside the pips."""
    rows = int(panel["rows"])
    pitch = float(panel["row_pitch"])
    top = int(panel["top"])
    right = int(panel["left"]) - 2
    left_edge = max(0, right - int(pitch * 6))
    lefts: List[int] = []
    widths: List[int] = []
    for frame in frames[:3]:
        for r in range(0, rows, max(1, rows // 8)):
            y = int(top + r * pitch - pitch * 0.2)
            band = (left_edge, max(0, y), max(8, right - left_edge), int(pitch))
            for bx, _by, bw, _bh in _text_boxes(frame, band, min_h=max(6, int(pitch * 0.25))):
                lefts.append(bx)
                widths.append(bw)
    if not lefts:
        guess = int(pitch * 2.2)
        return guess, max(0, right - guess)
    tag_left = int(np.percentile(lefts, 10))
    # run the tag box right up to the first pip: a clipped tag cannot be read,
    # and the blank-ROI gate makes the spare background cost nothing
    width = max(24, right - tag_left + 4)
    return width, tag_left


def _text_boxes(frame: np.ndarray, region: Tuple[int, int, int, int],
                min_h: int = 12, max_h: int = 70) -> List[Tuple[int, int, int, int]]:
    """Word-sized bright blobs inside a region, merged left to right."""
    x0, y0, w, h = region
    sub = frame[y0:y0 + h, x0:x0 + w]
    if sub.size == 0:
        return []
    g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    parts = []
    for i in range(1, n):
        x, y, cw, ch, a = stats[i]
        if min_h <= ch <= max_h and a >= 12 and cw <= w * 0.6:
            parts.append((x, y, cw, ch))
    parts.sort(key=lambda r: (r[1] // max(1, min_h), r[0]))
    merged: List[List[int]] = []
    for x, y, cw, ch in parts:
        placed = False
        for m in merged:
            same_line = abs((y + ch / 2) - (m[1] + m[3] / 2)) <= 0.6 * max(ch, m[3])
            near = x - (m[0] + m[2]) <= 0.9 * max(ch, m[3])
            if same_line and near and x >= m[0]:
                m[2] = max(m[0] + m[2], x + cw) - m[0]
                m[1] = min(m[1], y)
                m[3] = max(m[1] + m[3], y + ch) - m[1]
                placed = True
                break
        if not placed:
            merged.append([x, y, cw, ch])
    return [(x0 + m[0], y0 + m[1], m[2], m[3]) for m in merged]


def _merge_boxes(cands: Dict[Tuple[int, int, int, int], List[str]],
                 tol: int = 24) -> List[Tuple[Tuple[int, int, int, int], List[str]]]:
    """One candidate per place on screen, however the box wobbled per frame."""
    out: List[Tuple[List[int], List[str]]] = []
    for (x, y, w, h), reads in sorted(cands.items()):
        cx, cy = x + w / 2, y + h / 2
        hit = None
        for box, acc in out:
            if abs(cx - (box[0] + box[2] / 2)) <= tol and abs(cy - (box[1] + box[3] / 2)) <= tol:
                hit = (box, acc)
                break
        if hit is None:
            out.append(([x, y, w, h], list(reads)))
        else:
            box, acc = hit
            x1 = max(box[0] + box[2], x + w)
            y1 = max(box[1] + box[3], y + h)
            box[0], box[1] = min(box[0], x), min(box[1], y)
            box[2], box[3] = x1 - box[0], y1 - box[1]
            acc.extend(reads)
    return [((b[0], b[1], b[2], b[3]), acc) for b, acc in out]


def propose_counters(frames: Sequence[np.ndarray],
                     map_names: Sequence[str] = ()) -> Proposal:
    """Look along the top strip for small all-digit boxes and a map name.

    Which counter is which is decided by what they read, not by where they
    sit: the one that never exceeds the team count is the team counter, and
    the one that runs into three digits counts players. An organiser who puts
    them the other way round is then still read correctly.
    """
    p = Proposal()
    if not tesseract_available():
        p.note("tesseract is not installed, so text regions cannot be proposed")
        return p
    H, W = frames[0].shape[:2]
    strip = (0, 0, W, int(H * 0.16))
    votes: Dict[Tuple[int, int, int, int], List[str]] = {}
    for frame in frames[:5]:
        for box in _text_boxes(frame, strip):
            x, y, w, h = [int(v) for v in box]
            if w > W * 0.25 or w < 8:
                continue
            pad = (max(0, x - 4), max(0, y - 4), w + 8, h + 8)
            pre = prepare_roi(frame, pad, {"upscale": 3, "min_height": 60})
            if roi_is_blank(pre):
                continue
            txt, _conf = read_text([pre.get("binary_otsu"), pre["binary"]], ALNUM)
            if txt:
                votes.setdefault((x, y, w, h), []).append(txt)

    numeric: List[Tuple[Tuple[int, int, int, int], int, int]] = []   # box, maxval, maxlen
    named: List[Tuple[Tuple[int, int, int, int], str]] = []
    for box, reads in _merge_boxes(votes):
        clean = [re.sub(r"[^0-9]", "", r) for r in reads]
        hits = [c for c in clean if c]
        if hits and len(hits) >= max(1, len(reads) // 2) and all(len(c) <= 3 for c in hits):
            numeric.append((box, max(int(c) for c in hits), max(len(c) for c in hits)))
        elif map_names:
            from .ocr import fuzzy_pick
            pick, score = fuzzy_pick(reads[0], list(map_names))
            if pick and score >= 0.7:
                named.append((box, pick))

    teams = [c for c in numeric if c[1] <= 25 and c[2] <= 2]
    players = [c for c in numeric if c not in teams]
    if teams:
        box, _v, n = max(teams, key=lambda c: c[0][0])
        p.fields["teams_alive"] = {"box": [box[0] - 4, box[1] - 4, box[2] + 8, box[3] + 8],
                                   "kind": "numeral", "digits": max(2, n), "min": 1,
                                   "max": 25, "reset_at": 20}
        p.note(f"team counter at ({box[0]},{box[1]}): never read above 25")
    if players:
        box, v, n = max(players, key=lambda c: c[1])
        p.fields["players_alive"] = {"box": [box[0] - 4, box[1] - 4, box[2] + 8, box[3] + 8],
                                     "kind": "numeral", "digits": max(2, n), "min": 1,
                                     "max": 100, "reset_at": 70}
        p.note(f"player counter at ({box[0]},{box[1]}): read up to {v}")
    if named:
        box, nm = named[0]
        p.fields["map_name"] = {"box": [box[0] - 6, box[1] - 6, box[2] + 12, box[3] + 12],
                                "kind": "text"}
        p.note(f"map name read as {nm} at ({box[0]},{box[1]})")
    p.confidence = min(0.8, 0.25 * len(p.fields))
    if not p.fields:
        p.note("no all-digit boxes found along the top strip")
    return p


def propose_kill_feed(frames: Sequence[np.ndarray]) -> Proposal:
    """Find a stack of similar text lines: the feed is the only one that moves."""
    p = Proposal()
    H, W = frames[0].shape[:2]
    region = (int(W * 0.25), int(H * 0.04), int(W * 0.55), int(H * 0.45))
    best: Optional[Dict[str, Any]] = None
    for frame in frames:
        lines = [b for b in _text_boxes(frame, region) if b[2] >= W * 0.06]
        if len(lines) < 2:
            continue
        centres = sorted(b[1] + b[3] / 2 for b in lines)
        pitch = _median_gap(centres)
        if not pitch or pitch <= 0:
            continue
        on_pitch = [c for c in centres
                    if any(abs(abs(c - o) - pitch) < 0.3 * pitch for o in centres if o != c)]
        if len(on_pitch) < 2:
            continue
        rows = len(on_pitch)
        left = min(b[0] for b in lines)
        # a feed line is only as long as the names in it: leave room for longer ones
        width = int((max(b[0] + b[2] for b in lines) - left) * 1.45)
        top = int(min(on_pitch) - pitch * 0.45)
        cand = {"rows": rows, "pitch": round(pitch, 2), "left": left,
                "width": int(width), "top": max(0, top),
                "row_h": int(round(pitch * 0.8))}
        if best is None or rows > best["rows"]:
            best = cand
    if best is None:
        p.note("no evenly spaced stack of text lines found; the feed may be empty "
               "in the sampled frames")
        return p
    p.fields["kill_feed"] = {
        "box": [int(best["left"] - 6), int(best["top"]), int(best["width"] + 12),
                int(best["row_h"])],
        "kind": "feed", "rows": max(4, best["rows"]), "row_pitch": best["pitch"],
        "row_h": best["row_h"]}
    p.confidence = min(0.75, 0.3 + 0.12 * best["rows"])
    p.note(f"{best['rows']} feed lines on a {best['pitch']:.1f}px pitch")
    return p


def propose_hud(frames: Sequence[np.ndarray],
                map_names: Sequence[str] = ()) -> Proposal:
    out = Proposal()
    for part in (propose_team_panel(frames), propose_counters(frames, map_names),
                 propose_kill_feed(frames)):
        out.fields.update(part.fields)
        out.notes.extend(part.notes)
    out.confidence = round(min(1.0, 0.2 * len(out.fields)), 3)
    return out


# ---------------- the observer map ----------------
def _ring_extent(frames: Sequence[np.ndarray]
                 ) -> Optional[Tuple[float, float, float, float]]:
    """Bounding box of the zone rings, which nearly fill the map early on."""
    from .map_zones import fit_circle_ransac
    best: Optional[Tuple[float, float, float, float]] = None
    H, W = frames[0].shape[:2]
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for lo, hi in (((0, 0, 215), (179, 45, 255)), ((90, 110, 110), (120, 255, 255))):
            mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            keep = np.zeros_like(mask)
            for i in range(1, n):
                if max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) >= 0.2 * min(W, H):
                    keep[lab == i] = 255
            ys, xs = np.nonzero(keep)
            if len(xs) < 200:
                continue
            idx = np.random.default_rng(5).choice(len(xs), min(3000, len(xs)), replace=False)
            pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float64)
            fit = fit_circle_ransac(pts, 0.05 * min(W, H), 0.75 * min(W, H), tol_px=4.0)
            if fit is None:
                continue
            cx, cy, r, inl = fit
            if inl < 150:
                continue
            box = (cx - r, cy - r, cx + r, cy + r)
            if best is None or (box[2] - box[0]) > (best[2] - best[0]):
                best = box
    return best



def _modal_area(sizes: Sequence[int]) -> float:
    """The commonest blob size: one player.

    A mean or a median lands between one dot and the two-dot blobs a stacked
    squad makes, and sizing a cluster against that number miscounts the squad.
    The distribution has a clear peak at one dot, so take the peak.
    """
    a = np.array(sorted(sizes), dtype=float)
    if len(a) < 8:
        return float(np.median(a)) if len(a) else 60.0
    lo, hi = np.percentile(a, [5, 95])
    if hi <= lo:
        return float(np.median(a))
    hist, edges = np.histogram(a, bins=16, range=(lo, hi))
    i = int(np.argmax(hist))
    return float((edges[i] + edges[i + 1]) / 2.0)


def propose_map_area(frames: Sequence[np.ndarray]) -> Proposal:
    """The map square is where the player dots and the zone rings are."""
    p = Proposal()
    H, W = frames[0].shape[:2]
    pts: List[Tuple[float, float]] = []
    sizes: List[int] = []
    for frame in frames:
        blobs = _saturated_blobs(frame, sat_min=90, val_min=90, area=(12, 600),
                                 min_circularity=0.45)
        for x, y, w, h, a in blobs:
            pts.append((x + w / 2.0, y + h / 2.0))
            sizes.append(a)
    if len(pts) < 20:
        p.note("too few coloured dots to locate the map")
        return p
    xs = np.array([q[0] for q in pts])
    ys = np.array([q[1] for q in pts])
    # trim the extremes: a dot on the HUD outside the map should not stretch the box
    x0, x1 = np.percentile(xs, [1, 99])
    y0, y1 = np.percentile(ys, [1, 99])
    ring = _ring_extent(frames)
    if ring is not None:
        rx0, ry0, rx1, ry1 = ring
        x0, y0 = min(x0, rx0), min(y0, ry0)
        x1, y1 = max(x1, rx1), max(y1, ry1)
        p.note("widened to the zone ring, which reaches closer to the map edge "
               "than the players do")
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    pad = side * 0.06
    side = min(side + 2 * pad, min(W, H))
    bx = int(round(max(0, min(W - side, cx - side / 2))))
    by = int(round(max(0, min(H - side, cy - side / 2))))
    inside = float(np.mean((xs >= bx) & (xs <= bx + side) & (ys >= by) & (ys <= by + side)))
    p.fields["map_area"] = [bx, by, int(round(side)), int(round(side))]
    single = _modal_area(sizes)
    p.fields["dots"] = {"single_area": round(single, 1),
                        "min_area": round(max(8.0, single * 0.35), 1),
                        "max_area": round(single * 2.6, 1)}
    p.confidence = round(min(0.9, 0.3 + 0.6 * inside), 3)
    p.note(f"{len(pts)} dots across {len(frames)} frames, {inside * 100:.0f}% inside the "
           f"proposed square; one dot is about {single:.0f}px of area")
    return p
