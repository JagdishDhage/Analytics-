"""Map pixels to world metres, and world metres to a place a coach can name.

Two naming schemes are supported. A POI table (``maps/<map>.json``) gives real
names -- it is data, calibrated per map, and ships empty because inventing
coordinates for real BGMI locations would be inventing facts. A grid reference
(``D5``) needs no data at all and is derived from the map's world size, so it
always works and is what the reports fall back to.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class Poi:
    name: str
    x_m: float
    y_m: float
    radius_m: float = 200.0


@dataclass
class WorldMap:
    """The square map viewport and the world it stands for."""

    map_area: Tuple[int, int, int, int]          # x, y, w, h in frame pixels
    world_size_m: float
    map_name: Optional[str] = None
    pois: List[Poi] = None  # type: ignore[assignment]
    grid_cells: int = 8

    def __post_init__(self) -> None:
        if self.pois is None:
            self.pois = []

    # ---------------- transforms ----------------
    def to_world(self, px: float, py: float) -> Tuple[float, float]:
        x, y, w, h = self.map_area
        if w <= 0 or h <= 0:
            return (0.0, 0.0)
        return ((px - x) / w * self.world_size_m,
                (py - y) / h * self.world_size_m)

    def to_pixels(self, xm: float, ym: float) -> Tuple[float, float]:
        x, y, w, h = self.map_area
        return (x + xm / self.world_size_m * w, y + ym / self.world_size_m * h)

    def metres_per_pixel(self) -> float:
        _, _, w, _ = self.map_area
        return self.world_size_m / float(w) if w else 0.0

    def inside(self, xm: float, ym: float) -> bool:
        return 0.0 <= xm <= self.world_size_m and 0.0 <= ym <= self.world_size_m

    # ---------------- naming ----------------
    def grid_ref(self, xm: float, ym: float) -> str:
        step = self.world_size_m / self.grid_cells
        cx = min(self.grid_cells - 1, max(0, int(xm // step)))
        cy = min(self.grid_cells - 1, max(0, int(ym // step)))
        return f"{LETTERS[cx]}{cy + 1}"

    def nearest_poi(self, xm: float, ym: float) -> Tuple[Optional[Poi], float]:
        best, best_d = None, float("inf")
        for p in self.pois:
            d = math.dist((xm, ym), (p.x_m, p.y_m))
            if d < best_d:
                best, best_d = p, d
        return best, best_d

    def place(self, xm: float, ym: float) -> str:
        """A named POI when the position is inside one, else a grid reference."""
        poi, d = self.nearest_poi(xm, ym)
        if poi is not None and d <= poi.radius_m:
            return poi.name
        return self.grid_ref(xm, ym)

    # ---------------- loading ----------------
    @staticmethod
    def load_pois(path: str | Path) -> Tuple[List[Poi], Optional[str], Optional[float]]:
        d = json.loads(Path(path).read_text())
        pois = [Poi(p["name"].upper(), float(p["x_m"]), float(p["y_m"]),
                    float(p.get("radius_m", 200.0))) for p in d.get("pois", [])]
        return pois, d.get("map"), d.get("world_size_m")

    @staticmethod
    def save_pois(path: str | Path, map_name: str, world_size_m: float,
                  pois: Sequence[Poi]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "map": map_name, "world_size_m": world_size_m,
            "pois": [{"name": p.name, "x_m": round(p.x_m, 1), "y_m": round(p.y_m, 1),
                      "radius_m": round(p.radius_m, 1)} for p in pois],
        }, indent=2) + "\n")


def dist_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.dist(a, b)


def bearing_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Compass-style bearing from a to b, 0 = north (-y), clockwise."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    return (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0


def compass(bearing: float) -> str:
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return names[int(((bearing + 22.5) % 360) // 45)]


def centroid(points: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    pts = [p for p in points if p is not None]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def spread_m(points: Sequence[Tuple[float, float]]) -> Optional[float]:
    """Largest pairwise distance -- how stretched a squad is."""
    pts = [p for p in points if p is not None]
    if len(pts) < 2:
        return None
    return max(math.dist(a, b) for i, a in enumerate(pts) for b in pts[i + 1:])
