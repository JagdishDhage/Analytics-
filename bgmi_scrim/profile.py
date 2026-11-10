"""Calibration profiles.

A profile is pure data: HUD region-of-interest boxes for one resolution and one
organiser overlay, plus thresholds. Nothing about HUD geometry is hardcoded in
the extraction code; it is all read from here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Roi:
    """One region of interest, in the profile's reference-resolution pixels."""

    name: str
    box: Tuple[int, int, int, int]  # x, y, w, h
    kind: str  # numeral | text | team_panel | feed | colour_state
    meta: Dict[str, Any] = field(default_factory=dict)

    def scaled(self, sx: float, sy: float) -> "Roi":
        x, y, w, h = self.box
        return Roi(
            self.name,
            (int(round(x * sx)), int(round(y * sy)), int(round(w * sx)), int(round(h * sy))),
            self.kind,
            _scale_meta(self.meta, sx, sy),
        )


def _scale_meta(meta: Dict[str, Any], sx: float, sy: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if k in ("row_h", "row_pitch", "slot_h", "tag_h", "slot_dy", "tag_dy"):
            out[k] = int(round(float(v) * sy))
        elif k in ("slot_w", "tag_w", "slot_pitch", "slot_x0", "tag_x"):
            out[k] = int(round(float(v) * sx))
        elif k == "slot_boxes":
            out[k] = [
                [int(round(b[0] * sx)), int(round(b[1] * sy)),
                 int(round(b[2] * sx)), int(round(b[3] * sy))]
                for b in v
            ]
        else:
            out[k] = v
    return out


@dataclass
class ColourState:
    """An HSV band used to classify a slot/zone indicator by colour."""

    name: str
    hsv_lo: Tuple[int, int, int]
    hsv_hi: Tuple[int, int, int]
    min_fraction: float = 0.12


@dataclass
class Profile:
    name: str
    resolution: Tuple[int, int]
    overlay: str = "generic"
    rois: Dict[str, Roi] = field(default_factory=dict)
    slot_states: Dict[str, ColourState] = field(default_factory=dict)
    zone_states: Dict[str, ColourState] = field(default_factory=dict)
    sampling: Dict[str, float] = field(default_factory=lambda: {
        "base_fps": 2.0, "burst_fps": 5.0, "burst_window_s": 3.0})
    confirm_samples: Dict[str, int] = field(default_factory=lambda: {"default": 3})
    confidence_threshold: float = 0.60
    preprocess: Dict[str, Any] = field(default_factory=lambda: {
        "upscale": 2, "clahe_clip": 2.0, "clahe_grid": 8,
        "adaptive_block": 31, "adaptive_c": 5, "invert": True})
    team_tags: List[str] = field(default_factory=list)
    our_team: Optional[str] = None
    map_names: List[str] = field(default_factory=lambda: [
        "ERANGEL", "MIRAMAR", "SANHOK", "VIKENDI", "LIVIK", "NUSA", "RONDO", "KARAKIN"])
    feed_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "knock": ["KNOCKED", "KNOCKED OUT", "KNOCK"],
        "kill": ["FINISHED", "KILLED", "ELIMINATED"],
    })
    dedup_window_s: float = 12.0
    fuse_window_s: float = 6.0
    extras: Dict[str, Any] = field(default_factory=dict)   # "_"-prefixed notes/flags

    # ---------------- (de)serialisation ----------------
    @staticmethod
    def load(path: str | Path) -> "Profile":
        data = json.loads(Path(path).read_text())
        return Profile.from_dict(data)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Profile":
        p = Profile(
            name=data.get("name", "unnamed"),
            resolution=tuple(data.get("resolution", [1920, 1080])),  # type: ignore[arg-type]
            overlay=data.get("overlay", "generic"),
        )
        for k, v in (data.get("rois") or {}).items():
            meta = {kk: vv for kk, vv in v.items() if kk not in ("box", "kind")}
            p.rois[k] = Roi(k, tuple(v["box"]), v.get("kind", "numeral"), meta)  # type: ignore[arg-type]
        for k, v in (data.get("slot_states") or {}).items():
            p.slot_states[k] = ColourState(k, tuple(v["hsv_lo"]), tuple(v["hsv_hi"]),  # type: ignore[arg-type]
                                           float(v.get("min_fraction", 0.12)))
        for k, v in (data.get("zone_states") or {}).items():
            p.zone_states[k] = ColourState(k, tuple(v["hsv_lo"]), tuple(v["hsv_hi"]),  # type: ignore[arg-type]
                                            float(v.get("min_fraction", 0.12)))
        for attr in ("sampling", "confirm_samples", "preprocess", "feed_keywords"):
            if data.get(attr):
                getattr(p, attr).update(data[attr])
        for attr in ("confidence_threshold", "team_tags", "our_team", "map_names",
                     "dedup_window_s", "fuse_window_s"):
            if data.get(attr) is not None:
                setattr(p, attr, data[attr])
        p.extras = {k: v for k, v in data.items() if k.startswith("_")}
        return p

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.extras,
            "name": self.name,
            "resolution": list(self.resolution),
            "overlay": self.overlay,
            "sampling": self.sampling,
            "confirm_samples": self.confirm_samples,
            "confidence_threshold": self.confidence_threshold,
            "preprocess": self.preprocess,
            "team_tags": self.team_tags,
            "our_team": self.our_team,
            "map_names": self.map_names,
            "feed_keywords": self.feed_keywords,
            "dedup_window_s": self.dedup_window_s,
            "fuse_window_s": self.fuse_window_s,
            "rois": {k: {"box": list(r.box), "kind": r.kind, **r.meta} for k, r in self.rois.items()},
            "slot_states": {k: {"hsv_lo": list(c.hsv_lo), "hsv_hi": list(c.hsv_hi),
                                 "min_fraction": c.min_fraction} for k, c in self.slot_states.items()},
            "zone_states": {k: {"hsv_lo": list(c.hsv_lo), "hsv_hi": list(c.hsv_hi),
                                 "min_fraction": c.min_fraction} for k, c in self.zone_states.items()},
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    # ---------------- helpers ----------------
    def for_frame_size(self, width: int, height: int) -> "Profile":
        """Return a copy with every ROI scaled to an actual frame size."""
        rw, rh = self.resolution
        if (rw, rh) == (width, height):
            return self
        sx, sy = width / float(rw), height / float(rh)
        out = Profile.from_dict(self.to_dict())
        out.rois = {k: r.scaled(sx, sy) for k, r in self.rois.items()}
        out.resolution = (width, height)
        return out

    @property
    def calibrated(self) -> bool:
        """False for a template profile whose boxes have not been set yet."""
        return bool(self.extras.get("_calibrated", True))

    def confirm_n(self, field_name: str) -> int:
        return int(self.confirm_samples.get(field_name, self.confirm_samples.get("default", 3)))
