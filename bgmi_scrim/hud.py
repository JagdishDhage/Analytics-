"""HUD detectors. Geometry and colours come from the profile, never from here."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .ocr import fuzzy_pick, read_lines, read_numeral, read_text
from .preprocess import colour_fraction, prepare_roi, roi_is_blank
from .profile import ColourState, Profile, Roi

ALIVE, KNOCKED, DEAD = "alive", "knocked", "dead"


@dataclass
class SlotReading:
    team_row: int
    slot: int
    state: Optional[str]
    confidence: float


@dataclass
class PanelReading:
    rows: int
    slots: List[SlotReading] = field(default_factory=list)
    tags: Dict[int, Tuple[str, float]] = field(default_factory=dict)

    def row_slots(self, row: int) -> List[SlotReading]:
        return [s for s in self.slots if s.team_row == row]


def _cfg(prof: Profile) -> Dict[str, Any]:
    return prof.preprocess


def _bins(pre: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """Both binarisations of an ROI, for the OCR layer to choose between."""
    # Otsu first: on flat HUD plates it reads better, so the adaptive pass is
    # usually skipped by the OCR layer's early accept.
    return [pre.get("binary_otsu"), pre["binary"]]  # type: ignore[list-item]


def read_counter(frame: np.ndarray, prof: Profile, name: str) -> Tuple[Optional[int], float]:
    roi = prof.rois.get(name)
    if roi is None:
        return None, 0.0
    pre = prepare_roi(frame, roi.box, _cfg(prof))
    if roi_is_blank(pre):
        return None, 0.0
    digits = int(roi.meta.get("digits", 3))
    val, conf = read_numeral(_bins(pre), digits)
    lo = roi.meta.get("min")
    hi = roi.meta.get("max")
    if val is not None and ((lo is not None and val < lo) or (hi is not None and val > hi)):
        return None, 0.0   # outside the field's declared range: not a reading
    return val, conf


def classify_colour(bgr: np.ndarray, states: Dict[str, ColourState]) -> Tuple[Optional[str], float]:
    """Classify an indicator patch by which HSV band dominates it."""
    scores: List[Tuple[str, float]] = []
    for name, st in states.items():
        frac = colour_fraction(bgr, st.hsv_lo, st.hsv_hi)
        scores.append((name, frac / max(st.min_fraction, 1e-6)))
    if not scores:
        return None, 0.0
    scores.sort(key=lambda s: -s[1])
    best, best_s = scores[0]
    second_s = scores[1][1] if len(scores) > 1 else 0.0
    if best_s < 1.0:
        return None, 0.0
    margin = (best_s - second_s) / best_s if best_s > 0 else 0.0
    conf = min(1.0, 0.45 + 0.35 * min(1.0, best_s / 2.0) + 0.35 * margin)
    return best, conf


def _slot_box(panel: Roi, row: int, slot: int) -> Tuple[int, int, int, int]:
    m = panel.meta
    x, y, _, _ = panel.box
    pitch = float(m["row_pitch"])
    sx0 = float(m["slot_x0"])
    spitch = float(m["slot_pitch"])
    return (x + int(round(sx0 + slot * spitch)),
            y + int(round(row * pitch + float(m.get("slot_dy", 0)))),
            max(1, int(round(float(m["slot_w"])))), max(1, int(round(float(m["slot_h"])))))


def _tag_box(panel: Roi, row: int) -> Tuple[int, int, int, int]:
    m = panel.meta
    x, y, _, _ = panel.box
    return (x + int(round(float(m.get("tag_x", 0)))),
            y + int(round(row * float(m["row_pitch"]) + float(m.get("tag_dy", 0)))),
            max(1, int(round(float(m["tag_w"])))),
            max(1, int(round(float(m.get("tag_h", m["slot_h"]))))))


def read_team_panel(frame: np.ndarray, prof: Profile, read_tags: bool = False) -> PanelReading:
    panel = prof.rois.get("team_panel")
    if panel is None:
        return PanelReading(0)
    rows = int(panel.meta.get("rows", 25))
    slots = int(panel.meta.get("slots", 4))
    out = PanelReading(rows)
    for r in range(rows):
        for s in range(slots):
            box = _slot_box(panel, r, s)
            patch = prepare_roi(frame, box, {"upscale": 1})["bgr"]
            state, conf = classify_colour(patch, prof.slot_states)
            out.slots.append(SlotReading(r, s, state, conf))
        if read_tags:
            pre = prepare_roi(frame, _tag_box(panel, r), _cfg(prof))
            if roi_is_blank(pre):
                continue
            txt, tconf = read_text(_bins(pre), retry_constrained=True)
            if txt:
                if prof.team_tags:
                    pick, score = fuzzy_pick(txt, prof.team_tags)
                    if pick and score >= float(panel.meta.get("tag_min_score", 0.6)):
                        out.tags[r] = (pick, max(tconf, score))
                    # a tag that does not match the roster is not voted at all:
                    # a bad vote moves a whole team's events onto a wrong tag
                    continue
                else:
                    out.tags[r] = (txt, max(tconf, 0.5))
    return out


def read_feed_lines(frame: np.ndarray, prof: Profile) -> List[Tuple[str, float]]:
    roi = prof.rois.get("kill_feed")
    if roi is None:
        return []
    rows = int(roi.meta.get("rows", 0))
    if rows:  # row-by-row reading is steadier than one psm-6 block
        x, y, w, _ = roi.box
        pitch = float(roi.meta["row_pitch"])
        h = max(1, int(round(float(roi.meta.get("row_h", pitch)))))
        def one(r: int) -> Tuple[str, float]:
            pre = prepare_roi(frame, (x, y + r * pitch, w, h), _cfg(prof))
            if roi_is_blank(pre):
                return ("", 0.0)
            return read_text(_bins(pre))

        with ThreadPoolExecutor(max_workers=min(4, rows)) as ex:
            reads = list(ex.map(one, range(rows)))
        return [(t, c) for (t, c) in reads if t]
    pre = prepare_roi(frame, roi.box, _cfg(prof))
    if roi_is_blank(pre):
        return []
    return read_lines(pre["binary"])


def read_zone_phase(frame: np.ndarray, prof: Profile) -> Tuple[Optional[int], float]:
    return read_counter(frame, prof, "zone_phase")


def read_zone_state(frame: np.ndarray, prof: Profile) -> Tuple[Optional[str], float]:
    roi = prof.rois.get("zone_state")
    if roi is None or not prof.zone_states:
        return None, 0.0
    patch = prepare_roi(frame, roi.box, {"upscale": 1})["bgr"]
    return classify_colour(patch, prof.zone_states)


def read_choice_text(frame: np.ndarray, prof: Profile, roi_name: str,
                     candidates: List[str], min_score: float = 0.6
                     ) -> Tuple[Optional[str], float]:
    roi = prof.rois.get(roi_name)
    if roi is None:
        return None, 0.0
    pre = prepare_roi(frame, roi.box, _cfg(prof))
    if roi_is_blank(pre):
        return None, 0.0
    txt, conf = read_text(_bins(pre))
    if not txt:
        return None, 0.0
    pick, score = fuzzy_pick(txt, candidates)
    if pick is None or score < min_score:
        return None, min(conf, 0.3)
    return pick, min(1.0, 0.3 + 0.7 * score)


def read_placement(frame: np.ndarray, prof: Profile) -> Tuple[Optional[int], float]:
    """Rank shown on the elimination card, e.g. '#7'."""
    roi = prof.rois.get("placement")
    if roi is None:
        return None, 0.0
    pre = prepare_roi(frame, roi.box, _cfg(prof))
    if roi_is_blank(pre):
        return None, 0.0
    val, conf = read_numeral(_bins(pre), int(roi.meta.get("digits", 2)))
    if val is None or not (1 <= val <= int(roi.meta.get("max", 25))):
        return None, 0.0
    return val, conf


def read_winner_tag(frame: np.ndarray, prof: Profile) -> Tuple[Optional[str], float]:
    roi = prof.rois.get("winner_banner")
    if roi is None:
        return None, 0.0
    pre = prepare_roi(frame, roi.box, _cfg(prof))
    if roi_is_blank(pre):
        return None, 0.0
    txt, conf = read_text(_bins(pre))
    if not txt:
        return None, 0.0
    if prof.team_tags:
        pick, score = fuzzy_pick(txt, prof.team_tags)
        if pick and score >= float(roi.meta.get("min_score", 0.6)):
            return pick, min(1.0, 0.3 + 0.7 * score)
        return None, min(conf, 0.3)
    return txt, max(conf, 0.5)
