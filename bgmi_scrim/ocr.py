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


#: tesseract itself aborts on some inputs (it has died with SIGFPE on a wide,
#: sparse feed row). One engine crash must not end a whole run, and it must
#: never become a value either: the call degrades to an unreadable region,
#: which the callers already treat as "no reading". Counted so a run can
#: report how often the engine failed rather than hide it.
_OCR_FAILURES = 0


def ocr_failures() -> int:
    return _OCR_FAILURES


def reset_ocr_failures() -> None:
    global _OCR_FAILURES
    _OCR_FAILURES = 0


def _tess(img: np.ndarray, psm: int, whitelist: str,
          engine_whitelist: bool = False) -> Tuple[str, float]:
    global _OCR_FAILURES
    if not tesseract_available():
        raise OcrUnavailable("tesseract binary not found")
    # Passing tessedit_char_whitelist to the LSTM engine (--oem 1) costs the
    # word boundaries and the confidence: the same kill-feed row reads as
    # "MDYXDELTAFINISHEDXSPXBRAVO" conf 0 with it, and as "MDYxDELTA" 56,
    # "FINISHED" 93, "XSPxBRAVO" 90 without. Text ROIs therefore constrain
    # the alphabet in Python, below, after the engine has had its say.
    #
    # Numerals are the opposite case and keep the engine-side whitelist.
    # Measured on gate run A: filtering digits in Python instead took
    # players_alive from 0.99 to 0.7667, with 70 confidently wrong reads,
    # because the engine offers a letter for a digit and the filter drops it,
    # turning "44" into "4". read_numeral scores its own per-digit crops and
    # does not depend on the confidence this option flattens.
    cfg = f"--psm {psm} --oem 1"
    if engine_whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    try:
        data = pytesseract.image_to_data(_pad(img), config=cfg, output_type=Output.DICT)
    except OcrUnavailable:
        raise
    except Exception:  # engine abort, bad exit status, unparsable TSV
        _OCR_FAILURES += 1
        return "", 0.0
    allowed = set(whitelist)
    chunks: List[str] = []
    confs: List[float] = []
    for txt, conf in zip(data.get("text", []), data.get("conf", [])):
        # upper-case before filtering: the separator in "MDYxDELTA" is a
        # lowercase x, and dropping it would weld the tag to the player name
        txt = "".join(c for c in (txt or "").strip().upper() if c in allowed)
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


def _split_wide_component(sub: np.ndarray, max_parts: int = 6) -> List[np.ndarray]:
    """Split a component that is too wide to be one digit.

    Neighbouring HUD digits touch after thresholding more often than not
    ("44" becomes one blob), and a component count that cannot see that is
    worse than no count at all: it makes a one-digit read of a two-digit
    number look fully corroborated. Wide components are cut at the valleys of
    the column ink profile, with a width guard so that a hollow digit such as
    0 or 8 is never cut in half.
    """
    h, w = sub.shape[:2]
    if h <= 0 or w <= 1.15 * h:
        return [sub]
    k = min(max_parts, max(2, int(round(w / (0.85 * h)))))
    ink = (255 - sub > 0).sum(axis=0).astype(np.int32)
    min_gap = max(3, int(0.40 * h))
    cuts: List[int] = []
    for _ in range(k - 1):
        best, best_v = None, None
        for x in range(min_gap, w - min_gap):
            if any(abs(x - c) < min_gap for c in cuts):
                continue
            v = int(ink[x])
            if best_v is None or v < best_v:
                best, best_v = x, v
        if best is None:
            break
        cuts.append(best)
    cuts.sort()
    if not cuts:
        return [sub]
    parts, prev = [], 0
    for c in cuts + [w]:
        if c - prev >= 2:
            parts.append(sub[:, prev:c])
        prev = c
    return parts or [sub]


def segment_digits(binary: np.ndarray, min_h_frac: float = 0.18,
                   max_boxes: int = 6) -> List[np.ndarray]:
    """Split a numeral ROI into left-to-right digit crops (dark text on white).

    Component height is judged against the tallest component, not against the
    ROI, so a generously drawn calibration box does not make every digit look
    too short to be a digit.
    """
    inv = 255 - binary
    n, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    h = binary.shape[0]
    cands = []
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if hh < min_h_frac * h or area < 8:
            continue
        if w > 0.9 * binary.shape[1] and hh > 0.9 * h:
            continue
        cands.append((x, y, w, hh))
    if not cands:
        return []
    hmax = max(c[3] for c in cands)
    boxes = [c for c in cands if c[3] >= 0.55 * hmax]
    boxes.sort(key=lambda b: b[0])
    crops: List[np.ndarray] = []
    for (x, y, w, hh) in boxes:
        crops.extend(_split_wide_component(binary[y:y + hh, x:x + w], max_boxes))
    return crops[:max_boxes]


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
    whole_txt, whole_conf = _tess(binary, 7, DIGITS, engine_whitelist=True)
    whole = re.sub(r"\D", "", whole_txt)[:max_digits]

    crops = segment_digits(binary, max_boxes=max_digits)
    nseg = len(crops)
    per_digit: List[str] = []
    per_conf: List[float] = []
    seg_complete = nseg > 0
    for crop in crops:
        if crop.size == 0:
            seg_complete = False
            continue
        crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        t, c = _tess(crop, 10, DIGITS, engine_whitelist=True)
        t = re.sub(r"\D", "", t)[:1]
        if t:
            per_digit.append(t)
            per_conf.append(c)
        else:
            # an unreadable glyph must shorten nothing: without this, a digit
            # psm 10 cannot read would silently turn 100 into 10
            seg_complete = False
    seg = "".join(per_digit)
    seg_conf = float(np.mean(per_conf)) if per_conf else 0.0
    if max_digits == 1 and not seg:
        t, c = _tess(binary, 10, DIGITS, engine_whitelist=True)
        t = re.sub(r"\D", "", t)[:1]
        if t:
            seg, seg_conf, nseg, seg_complete = t, c, max(nseg, 1), True

    cands: dict[str, float] = {}
    if whole:
        cands[whole] = whole_conf if (nseg == 0 or len(whole) == nseg) else whole_conf * 0.35
    if seg:
        full = seg_complete and len(seg) == nseg
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
    if (whole and seg and whole != seg and len(whole) == len(seg)
            and seg_complete and len(whole) == nseg):
        # both readers produced a full-length number and they disagree:
        # that is not knowledge, so it must fall below any sane threshold
        conf = min(conf, 0.45)
    return int(text), min(1.0, conf)


def read_text(img, whitelist: str = ALNUM, psm: int = 7,
              accept_conf: float = 0.85,
              label_roi: bool = False) -> Tuple[str, float]:
    """Read a text ROI from one or more binarisations, keeping the best read.

    `label_roi` says this is a short region known to hold one label, such as
    a team tag in the panel. The two kinds of region want opposite rules, as
    measured on gate run A and on the feed-only clip:

      ranking            feed-only clip           panel team_alive_count
      confidence first   1.0000/1.0000, 0 phantom 0.9600, TX unresolved
      length first       4 phantoms               1.0000, TX resolved

    A feed line is long, and a longer garbled read of it parses into an event
    that is not there. A tag is two to five characters, and the way it fails
    is by coming back truncated or empty, so there the longer read is the
    better one and an empty read is worth asking again for.
    """
    best: Tuple[str, float] = ("", 0.0)
    imgs = _as_images(img)
    for b in imgs:
        txt, conf = _tess(b, psm, whitelist)
        txt = txt.upper().strip()
        if not txt:
            continue
        rank = (len(txt), conf) if label_roi else (conf, len(txt))
        cur = (len(best[0]), best[1]) if label_roi else (best[1], len(best[0]))
        if rank > cur:
            best = (txt, conf)
        if conf >= accept_conf and len(txt) >= len(best[0]):
            break
    if best[0] or not label_roi:
        return best
    # Nothing readable, and the caller says this ROI holds something. On a
    # short, highlighted ROI -- our own team's row in the panel is drawn
    # differently from the rest -- the free engine offers glyphs outside the
    # alphabet and the filter above leaves an empty string, so TX's row went
    # unnamed and its alive count with it. Constrain the engine and ask once
    # more. The confidence it reports under the whitelist is flat, which
    # costs nothing here: the caller scores a tag by how well it matches the
    # roster, and an empty read scores nothing at all.
    #
    # Only a caller that knows the region must contain text may ask for this.
    # A kill-feed row that reads as nothing usually IS nothing -- the feed is
    # shorter than its five slots -- and squeezing characters out of it turns
    # empty rows into events.
    for b in imgs:
        txt, _ = _tess(b, psm, whitelist, engine_whitelist=True)
        txt = txt.upper().strip()
        if txt and len(txt) > len(best[0]):
            best = (txt, 0.0)
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
