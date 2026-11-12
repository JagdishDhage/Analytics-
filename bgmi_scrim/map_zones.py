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
from dataclasses import dataclass, field
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


@dataclass
class ZoneStabiliser:
    """Reject circle fits that cannot be what the zone just did.

    A single RANSAC fit occasionally locks onto the wrong ring, or onto an arc
    of something else, and one such fit moves a reported zone centre a
    kilometre. A real zone moves only when a phase is announced, so a fit that
    disagrees with the recent ones is held back until it repeats -- the same
    confirm-before-believing rule the HUD counters use.
    """

    keep: int = 5
    confirm: int = 2
    history: Dict[str, List[Circle]] = field(default_factory=dict)
    pending: Dict[str, List[Circle]] = field(default_factory=dict)
    rejected: int = 0

    def accept(self, ring: str, c: Optional[Circle]) -> Optional[Circle]:
        if c is None:
            return None
        hist = self.history.setdefault(ring, [])
        if not hist:
            hist.append(c)
            return c
        med = self._median(hist)
        move = math.dist((c.cx, c.cy), (med.cx, med.cy))
        dr = abs(c.r - med.r)
        tol_c = max(80.0, 0.25 * med.r)
        tol_r = max(80.0, 0.20 * med.r)
        if move <= tol_c and dr <= tol_r:
            self.pending.pop(ring, None)
            hist.append(c)
            del hist[:-self.keep]
            return c
        pend = self.pending.setdefault(ring, [])
        if pend and (math.dist((c.cx, c.cy), (pend[-1].cx, pend[-1].cy)) <= tol_c
                     and abs(c.r - pend[-1].r) <= tol_r):
            pend.append(c)
        else:
            pend[:] = [c]
        if len(pend) >= self.confirm:
            # it kept saying the same new thing: the zone really did change
            self.history[ring] = list(pend[-self.keep:])
            self.pending.pop(ring, None)
            return c
        self.rejected += 1
        return None

    @staticmethod
    def _median(cs: List[Circle]) -> Circle:
        xs = sorted(c.cx for c in cs)
        ys = sorted(c.cy for c in cs)
        rs = sorted(c.r for c in cs)
        m = len(cs) // 2
        return Circle(xs[m], ys[m], rs[m], 1.0)


def read_circles(frame: np.ndarray, prof: MapProfile, world: WorldMap
                 ) -> Dict[str, Optional[Circle]]:
    """Fit every configured circle band, in world metres."""
    out: Dict[str, Optional[Circle]] = {}
    _, _, w, _ = [int(v) for v in prof.map_area]
    mpp = world.metres_per_pixel()
    for name, band in prof.circles.items():
        pts, (ox, oy) = _band_points(frame, prof, band)
        # The last circle of a match is a few pixels across and honestly holds
        # only a few dozen points. A flat minimum throws it away and loses the
        # final phase, so the floor follows the smallest circle this band is
        # allowed to find.
        r_min_px = band.min_radius_frac * w
        need = max(30, min(band.min_points, int(1.2 * 2 * math.pi * r_min_px)))
        if len(pts) < need:
            out[name] = None
            continue
        fit = fit_circle_ransac(pts, r_min_px, band.max_radius_frac * w,
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
