"""Where the picture actually is inside the frame.

A capture is rarely the clean 1920x1080 a profile was drawn against. Streams
get letterboxed, pillarboxed, rescaled to 1600x900, or padded by a recorder
that kept a 4:3 source in a 16:9 file. Every one of those shifts the HUD
without changing the frame size, so a profile that only scales by resolution
reads the wrong pixels and reports nothing at all. This module finds the
content box so a profile can be fitted into it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class ContentBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    frame_size: Tuple[int, int] = (0, 0)

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def is_full_frame(self) -> bool:
        fw, fh = self.frame_size
        return self.x == 0 and self.y == 0 and self.w == fw and self.h == fh

    def describe(self) -> str:
        fw, fh = self.frame_size
        if self.is_full_frame:
            return f"the whole {fw}x{fh} frame"
        bars = []
        if self.x > 0 or self.x + self.w < fw:
            bars.append(f"{self.x}px left, {fw - self.x - self.w}px right")
        if self.y > 0 or self.y + self.h < fh:
            bars.append(f"{self.y}px top, {fh - self.y - self.h}px bottom")
        return (f"{self.w}x{self.h} at ({self.x},{self.y}) inside {fw}x{fh}"
                + (" — bars: " + "; ".join(bars) if bars else ""))


def detect_content_box(frames: Sequence[np.ndarray], dark: int = 18,
                       min_fraction: float = 0.35) -> ContentBox:
    """Find the picture inside the bars, using the brightest of several frames.

    A single frame can be dark at the edges because the game is dark there;
    taking the per-pixel maximum over frames spread through the video means
    only a genuinely constant bar stays dark.
    """
    if not frames:
        raise ValueError("no frames to inspect")
    h, w = frames[0].shape[:2]
    acc = np.zeros((h, w), np.uint8)
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        if g.shape[:2] != (h, w):
            continue
        acc = np.maximum(acc, g)
    lit = acc > dark
    rows = lit.mean(axis=1)
    cols = lit.mean(axis=0)

    def span(profile: np.ndarray) -> Tuple[int, int]:
        on = np.flatnonzero(profile >= min_fraction)
        if on.size == 0:
            return 0, len(profile)
        return int(on[0]), int(on[-1] + 1)

    y0, y1 = span(rows)
    x0, x1 = span(cols)
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    inside = float(lit[y0:y1, x0:x1].mean()) if bw > 1 and bh > 1 else 0.0
    area = (bw * bh) / float(w * h)
    conf = min(1.0, 0.35 + 0.45 * inside + 0.2 * area)
    return ContentBox(x0, y0, bw, bh, round(conf, 4), (w, h))


def sample_frames(path: str, n: int = 6) -> List[np.ndarray]:
    """n frames spread across the video, skipping the first and last second."""
    from . import sampler
    info = sampler.probe(path)
    if info.frame_count <= 0:
        return []
    lo = int(min(info.fps, info.frame_count - 1))
    hi = max(lo + 1, info.frame_count - lo)
    step = max(1, (hi - lo) // max(1, n))
    idx = [min(info.frame_count - 1, lo + i * step) for i in range(n)]
    return [f for _fi, _t, f in sampler.iter_frames(path, idx)]


def parse_box(text: str) -> Optional[Tuple[int, int, int, int]]:
    """'x,y,w,h' from the command line; None for 'auto' or an empty value."""
    if not text or text.lower() == "auto":
        return None
    parts = [p for p in text.replace(" ", "").split(",") if p]
    if len(parts) != 4:
        raise ValueError("content box must be x,y,w,h or 'auto'")
    return tuple(int(float(p)) for p in parts)  # type: ignore[return-value]
