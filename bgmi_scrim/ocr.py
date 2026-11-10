"""Numeral and text OCR with calibrated confidence.

Two opinions are taken on every numeral: a whole-ROI read (tesseract psm 7)
and a per-digit read of segmented connected components (psm 10). Agreement
raises confidence, disagreement lowers it, so that a flickering read falls
below the profile threshold and is reported as "unknown" instead of a guess.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:  # pytesseract is optional at import time so tests can stub it
    import pytesseract
    from pytesseract import Output
    _HAVE_TESS = True
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore
    _HAVE_TESS = False

# Tesseract's OpenMP threading fights with our own per-ROI parallelism and
# makes each call several times slower in a container; one thread per call.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

DIGITS = "0123456789"
ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#"


class OcrUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def tesseract_available() -> bool:
    if not _HAVE_TESS:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _pad(img: np.ndarray, px: int = 10) -> np.ndarray:
    return cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=255)


def _tess(img: np.ndarray, psm: int, whitelist: str) -> Tuple[str, float]:
    if not tesseract_available():
        raise OcrUnavailable("tesseract binary not found")
    cfg = f"--psm {psm} --oem 1 -c tessedit_char_whitelist={whitelist}"
    data = pytesseract.image_to_data(_pad(img), config=cfg, output_type=Output.DICT)
    chunks: List[str] = []
    confs: List[float] = []
    for txt, conf in zip(data.get("text", []), data.get("conf", [])):
        txt = (txt or "").strip()
        try:
            c = float(conf)
        except (TypeError, ValueError):
            c = -1.0
        if txt and c >= 0:
            chunks.append(txt)
            confs.append(c / 100.0)
    if not chunks:
        return "", 0.0
    return " ".join(chunks), float(np.mean(confs))


def segment_digits(binary: np.ndarray, min_h_frac: float = 0.35,
                   max_boxes: int = 6) -> List[np.ndarray]:
    """Split a numeral ROI into left-to-right digit crops (dark text on white)."""
    inv = 255 - binary
    n, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    h = binary.shape[0]
    boxes = []
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if hh < min_h_frac * h or area < 8:
            continue
        if w > 0.9 * binary.shape[1] and hh > 0.9 * h:
            continue
        boxes.append((x, y, w, hh))
    boxes.sort(key=lambda b: b[0])
    if len(boxes) > max_boxes:
        boxes = boxes[:max_boxes]
    return [binary[y:y + hh, x:x + w] for (x, y, w, hh) in boxes]


def _as_images(img) -> List[np.ndarray]:
    if isinstance(img, np.ndarray):
        return [img]
    return [i for i in img if isinstance(i, np.ndarray) and i.size]


def read_numeral(img, max_digits: int = 3, accept_conf: float = 0.88) -> Tuple[Optional[int], float]:
    """Read an integer from one or more binarisations of a preprocessed ROI.

    Binarisations are tried in order and the rest are skipped once one reads
    confidently (that is what keeps the per-frame cost down). When two do run
    and agree, the confidence is raised.
    """
    results = []
    for b in _as_images(img):
        r = _read_numeral_one(b, max_digits)
        results.append(r)
        if r[0] is not None and r[1] >= accept_conf:
            break
    results = [r for r in results if r[0] is not None]
    if not results:
        return None, 0.0
    vals = {}
    for v, c in results:
        vals.setdefault(v, []).append(c)
    if len(vals) == 1 and len(results) > 1:
        v = results[0][0]
        return v, min(1.0, max(c for _, c in results) + 0.1)
    return max(results, key=lambda r: r[1])


def _read_numeral_one(binary: np.ndarray, max_digits: int = 3) -> Tuple[Optional[int], float]:
    """Read an integer from one binarisation.

    Three pieces of evidence are combined: the whole-ROI read (psm 7), the
    per-digit read of segmented connected components (psm 10), and the number
    of components found, which is a font-independent count of how many digits
    are actually on screen. A value whose digit count disagrees with the
    component count is capped below any sane threshold, so it surfaces as
    "unknown" rather than as a confident wrong number.
    """
    whole_txt, whole_conf = _tess(binary, 7, DIGITS)
    whole = re.sub(r"\D", "", whole_txt)[:max_digits]

    crops = segment_digits(binary, max_boxes=max_digits)
    nseg = len(crops)
    per_digit: List[str] = []
    per_conf: List[float] = []
    for crop in crops:
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        t, c = _tess(crop, 10, DIGITS)
        t = re.sub(r"\D", "", t)[:1]
        if t:
            per_digit.append(t)
            per_conf.append(c)
    seg = "".join(per_digit)
    seg_conf = float(np.mean(per_conf)) if per_conf else 0.0
    if max_digits == 1 and not seg:
        t, c = _tess(binary, 10, DIGITS)
        t = re.sub(r"\D", "", t)[:1]
        if t:
            seg, seg_conf, nseg = t, c, max(nseg, 1)

    cands: dict[str, float] = {}
    if whole:
        cands[whole] = whole_conf if (nseg == 0 or len(whole) == nseg) else whole_conf * 0.35
    if seg:
        full = (len(seg) == nseg)
        score = seg_conf if full else seg_conf * 0.35
        cands[seg] = max(cands.get(seg, 0.0), score)
    if not cands:
        return None, 0.0
    best = max(cands.items(), key=lambda kv: kv[1])
    text, score = best
    agree = (whole and seg and whole == seg)
    conf = 0.35 + 0.65 * score if agree else 0.25 + 0.65 * score
    if nseg and len(text) != nseg:
        conf = min(conf, 0.45)   # digit count disagrees with what is on screen
    return int(text), min(1.0, conf)


def read_text(img, whitelist: str = ALNUM, psm: int = 7,
              accept_conf: float = 0.85) -> Tuple[str, float]:
    """Read a text ROI from one or more binarisations, keeping the best read."""
    best: Tuple[str, float] = ("", 0.0)
    for b in _as_images(img):
        txt, conf = _tess(b, psm, whitelist)
        txt = txt.upper().strip()
        if not txt:
            continue
        if (conf, len(txt)) > (best[1], len(best[0])):
            best = (txt, conf)
        if conf >= accept_conf:
            break
    return best


# OCR confusions seen on HUD fonts, used when matching against a known roster
_GLYPH_VARIANTS = {"0": "O", "O": "0", "1": "I", "I": "1", "L": "1",
                   "5": "S", "S": "5", "8": "B", "B": "8", "2": "Z", "Z": "2"}


def glyph_variants(text: str, limit: int = 8) -> List[str]:
    """A few spellings of an OCR string with confusable glyphs swapped."""
    out = [text]
    for i, ch in enumerate(text):
        alt = _GLYPH_VARIANTS.get(ch)
        if alt:
            out.append(text[:i] + alt + text[i + 1:])
        if len(out) >= limit:
            break
    swapped = "".join(_GLYPH_VARIANTS.get(c, c) for c in text)
    if swapped not in out:
        out.append(swapped)
    return out


def read_lines(binary: np.ndarray, whitelist: str = ALNUM) -> List[Tuple[str, float]]:
    """Read a multi-line block (kill feed) as a list of (line, confidence)."""
    if not tesseract_available():
        raise OcrUnavailable("tesseract binary not found")
    cfg = f"--psm 6 --oem 1 -c tessedit_char_whitelist={whitelist}"
    data = pytesseract.image_to_data(_pad(binary), config=cfg, output_type=Output.DICT)
    rows: dict[tuple, list] = {}
    for i, txt in enumerate(data.get("text", [])):
        txt = (txt or "").strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError, KeyError):
            conf = -1.0
        if not txt or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        rows.setdefault(key, []).append((txt, conf / 100.0, data["top"][i]))
    out: List[Tuple[str, float]] = []
    for key in sorted(rows, key=lambda k: min(w[2] for w in rows[k])):
        words = rows[key]
        out.append((" ".join(w[0] for w in words).upper(),
                    float(np.mean([w[1] for w in words]))))
    return out


def fuzzy_pick(text: str, candidates: Sequence[str]) -> Tuple[Optional[str], float]:
    """Pick the closest candidate to an OCR string; score in [0, 1]."""
    import difflib
    if not text or not candidates:
        return None, 0.0
    best, best_score = None, 0.0
    for variant in glyph_variants(text):
        for cand in candidates:
            score = difflib.SequenceMatcher(None, variant, cand.upper()).ratio()
            if variant != text:
                score *= 0.98   # a glyph swap is evidence, but weaker
            if score > best_score:
                best, best_score = cand, score
    return best, best_score
