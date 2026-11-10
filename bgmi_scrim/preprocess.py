"""ROI preprocessing: crop, 2x upscale, CLAHE, adaptive threshold."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


def crop(frame: np.ndarray, box: Sequence[int]) -> np.ndarray:
    x, y, w, h = [int(v) for v in box]
    H, W = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((1, 1, 3), dtype=frame.dtype)
    return frame[y0:y1, x0:x1]


def clahe(gray: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    op = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    return op.apply(gray)


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


def prepare_roi(frame: np.ndarray, box: Sequence[int],
                cfg: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
    """Crop -> upscale -> CLAHE -> adaptive threshold.

    Returns the intermediate images so detectors can pick the one they need
    (colour matching wants ``bgr``, OCR wants ``binary``).
    """
    cfg = cfg or {}
    up = int(cfg.get("upscale", 2))
    bgr = crop(frame, box)
    # small HUD ROIs need more than 2x before OCR can see the strokes
    min_h = int(cfg.get("min_height", 0))
    if min_h and bgr.shape[0] * up < min_h:
        up = max(up, int(np.ceil(min_h / max(1, bgr.shape[0]))))
    if up != 1:
        bgr = cv2.resize(bgr, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    eq = clahe(gray, cfg.get("clahe_clip", 2.0), cfg.get("clahe_grid", 8))
    block = max(3, _odd(int(cfg.get("adaptive_block", 31))))
    c = float(cfg.get("adaptive_c", 5))
    binary = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block, c)
    _, otsu = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cfg.get("invert", True):
        # HUD text is light on dark; OCR wants dark text on white.
        binary = 255 - binary
        otsu = 255 - otsu
    binary = cv2.medianBlur(binary, 3)
    otsu = cv2.medianBlur(otsu, 3)
    # Adaptive threshold handles gradient overlays; Otsu handles small ROIs on a
    # flat plate, where adaptive turns the flat background into speckle. Both are
    # produced and the OCR layer keeps whichever reads with more confidence.
    return {"bgr": bgr, "gray": gray, "eq": eq, "binary": binary, "binary_otsu": otsu,
            "upscale": up, "contrast": float(gray.std())}


def roi_is_blank(pre: Dict[str, Any], min_contrast: float = 9.0) -> bool:
    """True when an ROI holds no glyph at all (empty kill-feed row, no banner).

    Thresholding a flat patch produces noise that OCR will happily spend
    seconds on, so blank ROIs are gated out before OCR, not after.
    """
    if float(pre.get("contrast", 0.0)) < min_contrast:
        return True
    otsu = pre.get("binary_otsu")
    if otsu is None:
        return False
    ink = 1.0 - (float(np.count_nonzero(otsu)) / float(otsu.size))
    return ink < 0.004 or ink > 0.6


def colour_fraction(bgr: np.ndarray, hsv_lo: Sequence[int], hsv_hi: Sequence[int]) -> float:
    """Fraction of pixels of an ROI inside an HSV band (hue wrap handled)."""
    if bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo = np.array(hsv_lo, dtype=np.int16)
    hi = np.array(hsv_hi, dtype=np.int16)
    if lo[0] <= hi[0]:
        mask = cv2.inRange(hsv, lo.astype(np.uint8), hi.astype(np.uint8))
    else:  # hue wraps around 180 (red)
        m1 = cv2.inRange(hsv, np.array([lo[0], lo[1], lo[2]], np.uint8),
                         np.array([179, hi[1], hi[2]], np.uint8))
        m2 = cv2.inRange(hsv, np.array([0, lo[1], lo[2]], np.uint8),
                         np.array([hi[0], hi[1], hi[2]], np.uint8))
        mask = cv2.bitwise_or(m1, m2)
    return float(np.count_nonzero(mask)) / float(mask.size)


def match_template(haystack_gray: np.ndarray, needle_gray: np.ndarray) -> Tuple[float, Tuple[int, int]]:
    """Normalised-correlation template match. Returns (score, top-left)."""
    if (haystack_gray.shape[0] < needle_gray.shape[0]
            or haystack_gray.shape[1] < needle_gray.shape[1]):
        return 0.0, (0, 0)
    res = cv2.matchTemplate(haystack_gray, needle_gray, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    return float(mx), (int(loc[0]), int(loc[1]))
