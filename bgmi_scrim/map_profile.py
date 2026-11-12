"""Calibration profile for the map stream (phase 3).

Separate from the spectator-HUD profile because it describes a different
picture: the full-screen observer map, where every player is a labelled dot
and the zone is a pair of circles. Like the HUD profile it is pure data --
the map viewport box, the dot and label geometry, the colour bands for the
zone circles, the world size of the map in metres, and the roster.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: BGMI map sizes in metres. Battle-royale maps are square; the value is the
#: edge length used to turn map pixels into metres. Override per profile when
#: an organiser's map view crops or letterboxes the playable area.
WORLD_SIZE_M = {
    "ERANGEL": 8000, "MIRAMAR": 8000, "RONDO": 8000, "VIKENDI": 6000,
    "SANHOK": 4000, "LIVIK": 2000, "KARAKIN": 2000, "NUSA": 1000,
}

DEFAULT_LABEL_PATTERN = r"(?P<team>[A-Z0-9]{2,8})\s*[X×_\-\s]\s*(?P<player>[A-Z0-9]{2,12})"


@dataclass
class CircleBand:
    name: str
    hsv_lo: Tuple[int, int, int]
    hsv_hi: Tuple[int, int, int]
    min_radius_frac: float = 0.02
    max_radius_frac: float = 0.95
    min_points: int = 120
    large_components_only: bool = True
    tol_px: float = 4.0


@dataclass
class MapProfile:
    name: str = "unnamed_map_profile"
    resolution: Tuple[int, int] = (1920, 1080)
    overlay: str = "generic"
    map_name: Optional[str] = None
    world_size_m: Optional[float] = None
    map_area: Tuple[int, int, int, int] = (0, 0, 1080, 1080)   # x, y, w, h
    pois_file: Optional[str] = None

    dots: Dict[str, Any] = field(default_factory=lambda: {
        "min_area": 14, "max_area": 400, "sat_min": 90, "val_min": 90,
        "single_area": 60.0, "cluster_ratio": 1.6, "max_cluster": 4,
        "dedup_px": 3.0,
    })
    label: Dict[str, Any] = field(default_factory=lambda: {
        "dx": 10, "dy": -20, "w": 150, "h": 20, "min_score": 0.6, "max_per_frame": 120,
    })
    circles: Dict[str, CircleBand] = field(default_factory=dict)
    sampling: Dict[str, float] = field(default_factory=lambda: {
        "base_fps": 2.0, "label_refresh_s": 10.0, "label_prepass_frames": 4,
    })
    tracking: Dict[str, float] = field(default_factory=lambda: {
        "max_speed_mps": 140.0, "max_miss_s": 8.0, "min_track_s": 2.0,
        "gate_floor_m": 60.0,
    })
    roster: Dict[str, List[str]] = field(default_factory=dict)
    label_pattern: str = DEFAULT_LABEL_PATTERN
    confidence_threshold: float = 0.6
    preprocess: Dict[str, Any] = field(default_factory=lambda: {
        "upscale": 2, "min_height": 56, "clahe_clip": 2.0, "clahe_grid": 8,
        "adaptive_block": 25, "adaptive_c": 5, "invert": True,
    })
    extras: Dict[str, Any] = field(default_factory=dict)

    # ---------------- (de)serialisation ----------------
    @staticmethod
    def load(path: str | Path) -> "MapProfile":
        return MapProfile.from_dict(json.loads(Path(path).read_text()))

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MapProfile":
        p = MapProfile(
            name=d.get("name", "unnamed_map_profile"),
            resolution=tuple(d.get("resolution", [1920, 1080])),  # type: ignore[arg-type]
            overlay=d.get("overlay", "generic"),
            map_name=(d.get("map") or {}).get("name"),
            world_size_m=(d.get("map") or {}).get("world_size_m"),
            pois_file=(d.get("map") or {}).get("pois_file"),
        )
        if d.get("map_area"):
            p.map_area = tuple(d["map_area"])  # type: ignore[assignment]
        for attr in ("dots", "label", "sampling", "tracking", "preprocess"):
            if d.get(attr):
                getattr(p, attr).update(d[attr])
        for k, v in (d.get("circles") or {}).items():
            p.circles[k] = CircleBand(k, tuple(v["hsv_lo"]), tuple(v["hsv_hi"]),  # type: ignore[arg-type]
                                      float(v.get("min_radius_frac", 0.02)),
                                      float(v.get("max_radius_frac", 0.95)),
                                      int(v.get("min_points", 120)),
                                      bool(v.get("large_components_only", True)),
                                      float(v.get("tol_px", 4.0)))
        for attr in ("roster", "label_pattern", "confidence_threshold"):
            if d.get(attr) is not None:
                setattr(p, attr, d[attr])
        p.extras = {k: v for k, v in d.items() if k.startswith("_")}
        if p.world_size_m is None and p.map_name:
            p.world_size_m = WORLD_SIZE_M.get(p.map_name.upper())
        return p

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.extras,
            "name": self.name,
            "resolution": list(self.resolution),
            "overlay": self.overlay,
            "map": {"name": self.map_name, "world_size_m": self.world_size_m,
                    "pois_file": self.pois_file},
            "map_area": list(self.map_area),
            "dots": self.dots,
            "label": self.label,
            "circles": {k: {"hsv_lo": list(c.hsv_lo), "hsv_hi": list(c.hsv_hi),
                            "min_radius_frac": c.min_radius_frac,
                            "max_radius_frac": c.max_radius_frac,
                            "min_points": c.min_points,
                            "large_components_only": c.large_components_only,
                            "tol_px": c.tol_px}
                       for k, c in self.circles.items()},
            "sampling": self.sampling,
            "tracking": self.tracking,
            "roster": self.roster,
            "label_pattern": self.label_pattern,
            "confidence_threshold": self.confidence_threshold,
            "preprocess": self.preprocess,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    # ---------------- helpers ----------------
    @property
    def calibrated(self) -> bool:
        return bool(self.extras.get("_calibrated", True))

    @property
    def teams(self) -> List[str]:
        return sorted(self.roster)

    def players_of(self, team: str) -> List[str]:
        return list(self.roster.get(team, []))

    def all_labels(self) -> List[Tuple[str, str]]:
        return [(t, p) for t in sorted(self.roster) for p in self.roster[t]]

    def label_re(self) -> "re.Pattern[str]":
        return re.compile(self.label_pattern, re.IGNORECASE)

    def fitted(self, frame_w: int, frame_h: int,
               content_box: Optional[Tuple[int, int, int, int]] = None) -> "MapProfile":
        """Scale to the picture, then place it where the picture is."""
        if content_box is None:
            return self.for_frame_size(frame_w, frame_h)
        cx, cy, cw, ch = [int(v) for v in content_box]
        out = self.for_frame_size(cw, ch)
        x, y, w, h = out.map_area
        out.map_area = (x + cx, y + cy, w, h)
        out.resolution = (frame_w, frame_h)
        return out

    def for_frame_size(self, width: int, height: int) -> "MapProfile":
        rw, rh = self.resolution
        if (rw, rh) == (width, height):
            return self
        sx, sy = width / float(rw), height / float(rh)
        out = MapProfile.from_dict(self.to_dict())
        x, y, w, h = self.map_area
        out.map_area = (int(round(x * sx)), int(round(y * sy)),
                        int(round(w * sx)), int(round(h * sy)))
        area_scale = sx * sy
        for k in ("min_area", "max_area", "single_area"):
            if k in out.dots:
                out.dots[k] = float(out.dots[k]) * area_scale
        for k, s in (("dx", sx), ("dy", sy), ("w", sx), ("h", sy)):
            out.label[k] = float(self.label[k]) * s
        out.dots["dedup_px"] = float(self.dots.get("dedup_px", 3.0)) * sx
        out.resolution = (width, height)
        return out
