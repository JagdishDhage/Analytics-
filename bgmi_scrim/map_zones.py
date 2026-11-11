"""Reading the zone circles off the map.

The safe circle and the blue circle are the only things on the map that are
large, thin and round. They are fitted with RANSAC rather than least squares
because map text, icons and the player dots land in the same colour mask, and
one stray cluster of white pixels is enough to drag a least-squares circle
somewhere it has never been.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .map_profile import CircleBand, MapProfile
from .map_world import WorldMap


@dataclass
class Circle:
    cx: float          # world metres
    cy: float
    r: float
    confidence: float
    inliers: int = 0
    points: int = 0


def _band_points(frame: np.ndarray, prof: MapProfile, band: CircleBand,
                 max_points: int = 4000) -> Tuple[np.ndarray, Tuple[int, int]]:
    x, y, w, h = [int(v) for v in prof.map_area]
    sub = frame[max(0, y):y + h, max(0, x):x + w]
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    lo = np.array(band.hsv_lo, np.uint8)
    hi = np.array(band.hsv_hi, np.uint8)
    if band.hsv_lo[0] <= band.hsv_hi[0]:
        mask = cv2.inRange(hsv, lo, hi)
    else:
        m1 = cv2.inRange(hsv, np.array([band.hsv_lo[0], lo[1], lo[2]], np.uint8),
                         np.array([179, hi[1], hi[2]], np.uint8))
        m2 = cv2.inRange(hsv, np.array([0, lo[1], lo[2]], np.uint8), hi)
        mask = cv2.bitwise_or(m1, m2)
    if band.large_components_only:
        # Place names and player labels are the same colour as the safe
        # circle. A glyph is a small blob; the ring is one component as wide
        # as the circle, so keeping only large components leaves the ring.
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros_like(mask)
        min_span = max(8, int(band.min_radius_frac * mask.shape[1] * 1.2))
        for i in range(1, n):
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if max(bw, bh) >= min_span:
                keep[lab == i] = 255
        mask = keep
    ys, xs = np.nonzero(mask)
    if len(xs) > max_points:
        idx = np.random.default_rng(12345).choice(len(xs), max_points, replace=False)
        xs, ys = xs[idx], ys[idx]
    return np.stack([xs, ys], axis=1).astype(np.float64), (x, y)


def _circle_through(p1, p2, p3) -> Optional[Tuple[float, float, float]]:
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-6:
        return None
    ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay)
          + (cx ** 2 + cy ** 2) * (ay - by)) / d
    uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx)
          + (cx ** 2 + cy ** 2) * (bx - ax)) / d
    return ux, uy, math.dist((ux, uy), (ax, ay))


def fit_circle_ransac(points: np.ndarray, r_min: float, r_max: float,
                      tol_px: float = 2.5, iters: int = 400,
                      seed: int = 7) -> Optional[Tuple[float, float, float, int]]:
    """Returns (cx, cy, r, inliers) in the same pixel space as ``points``."""
    n = len(points)
    if n < 12:
        return None
    rng = random.Random(seed)
    best = None
    best_in = 0
    for _ in range(iters):
        i, j, k = rng.randrange(n), rng.randrange(n), rng.randrange(n)
        if i == j or j == k or i == k:
            continue
        c = _circle_through(points[i], points[j], points[k])
        if c is None:
            continue
        cx, cy, r = c
        if not (r_min <= r <= r_max):
            continue
        d = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - r)
        inl = int(np.count_nonzero(d <= tol_px))
        if inl > best_in:
            best_in, best = inl, (cx, cy, r)
    if best is None:
        return None
    # refine on the inliers (algebraic least squares)
    cx, cy, r = best
    d = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - r)
    pts = points[d <= tol_px]
    if len(pts) >= 3:
        A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
        b = (pts[:, 0] ** 2 + pts[:, 1] ** 2)
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            cx, cy = float(sol[0]), float(sol[1])
            r = float(math.sqrt(max(0.0, sol[2] + cx * cx + cy * cy)))
        except np.linalg.LinAlgError:
            pass
    return cx, cy, r, best_in


def read_circles(frame: np.ndarray, prof: MapProfile, world: WorldMap
                 ) -> Dict[str, Optional[Circle]]:
    """Fit every configured circle band, in world metres."""
    out: Dict[str, Optional[Circle]] = {}
    _, _, w, _ = [int(v) for v in prof.map_area]
    mpp = world.metres_per_pixel()
    for name, band in prof.circles.items():
        pts, (ox, oy) = _band_points(frame, prof, band)
        if len(pts) < band.min_points:
            out[name] = None
            continue
        fit = fit_circle_ransac(pts, band.min_radius_frac * w, band.max_radius_frac * w,
                                tol_px=band.tol_px)
        if fit is None:
            out[name] = None
            continue
        cx, cy, r, inl = fit
        ratio = inl / float(len(pts))
        conf = max(0.0, min(1.0, 0.25 + 0.75 * ratio))
        wx, wy = world.to_world(ox + cx, oy + cy)
        out[name] = Circle(wx, wy, r * mpp, conf, inliers=inl, points=len(pts))
    return out
