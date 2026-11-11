"""Calibrating and verifying a map-stream profile."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import sampler
from .map_dots import ColourBook, assign_teams_by_colour, detect_dots, read_dot_labels
from .map_profile import MapProfile
from .map_world import Poi, WorldMap
from .map_zones import read_circles


def grab_frame(video: str, at: float = 90.0) -> Tuple[np.ndarray, sampler.VideoInfo]:
    info = sampler.probe(video)
    idx = min(max(0, int(at * info.fps)), max(0, info.frame_count - 1))
    for _, _, frame in sampler.iter_frames(video, [idx]):
        return frame, info
    raise IOError(f"could not read a frame from {video}")


def _gui() -> bool:
    try:
        cv2.namedWindow("probe", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("probe")
        return True
    except Exception:
        return False


def template(video: str, out: str, base: str, map_name: Optional[str] = None) -> MapProfile:
    info = sampler.probe(video)
    prof = MapProfile.load(base).for_frame_size(info.width, info.height)
    prof.name = Path(out).stem
    prof.resolution = (info.width, info.height)
    if map_name:
        from .map_profile import WORLD_SIZE_M
        prof.map_name = map_name.upper()
        prof.world_size_m = WORLD_SIZE_M.get(prof.map_name, prof.world_size_m)
    prof.save(out)
    print(f"wrote {out} ({info.width}x{info.height}, scaled from {base})")
    return prof


def interactive(video: str, out: str, base: Optional[str] = None, at: float = 90.0,
                map_name: Optional[str] = None) -> MapProfile:
    frame, info = grab_frame(video, at)
    prof = (MapProfile.load(base).for_frame_size(info.width, info.height) if base
            else MapProfile(name=Path(out).stem, resolution=(info.width, info.height)))
    prof.name = Path(out).stem
    prof.resolution = (info.width, info.height)
    if map_name:
        from .map_profile import WORLD_SIZE_M
        prof.map_name = map_name.upper()
        prof.world_size_m = WORLD_SIZE_M.get(prof.map_name, prof.world_size_m)
    if not _gui():
        raise RuntimeError(
            "OpenCV highgui is not available here. Use 'mapcalibrate --template' to scale "
            "an existing map profile, edit map_area in the JSON, then check it with "
            "'mapcalibrate --preview'.")
    print("Drag a box around the PLAYABLE MAP SQUARE (not the window), then ENTER.")
    box = cv2.selectROI("map area", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("map area")
    if box[2] > 0 and box[3] > 0:
        prof.map_area = tuple(int(v) for v in box)  # type: ignore[assignment]
    prof.save(out)
    print(f"wrote {out}: map_area={prof.map_area}")
    print("Dot sizes, label offsets and the circle colour bands are edited in the JSON "
          "and checked with --preview.")
    return prof


def pick_pois(video: str, profile_path: str, pois_path: str, at: float = 90.0) -> None:
    """Click places on the map and name them, writing a POI table."""
    frame, info = grab_frame(video, at)
    prof = MapProfile.load(profile_path).for_frame_size(info.width, info.height)
    world = WorldMap(tuple(int(v) for v in prof.map_area),  # type: ignore[arg-type]
                     float(prof.world_size_m or 8000.0), prof.map_name)
    if not _gui():
        raise RuntimeError(
            "OpenCV highgui is not available here. Write the POI table by hand; "
            "maps/README.md documents the format, and 'mapcalibrate --preview' prints "
            "the world coordinates under the map grid so values can be read off.")
    pois: List[Poi] = []
    if Path(pois_path).exists():
        pois, _, _ = WorldMap.load_pois(pois_path)
    canvas = frame.copy()

    def on_click(event: int, x: int, y: int, flags: int, _param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        xm, ym = world.to_world(x, y)
        name = input(f"name for ({xm:.0f}, {ym:.0f}) m [blank to skip]: ").strip().upper()
        if not name:
            return
        pois.append(Poi(name, xm, ym, 220.0))
        cv2.circle(canvas, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(canvas, name, (x + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("pois", canvas)

    cv2.namedWindow("pois", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("pois", on_click)
    cv2.imshow("pois", canvas)
    print("Click a place, type its name in this terminal. ESC when done.")
    while cv2.waitKey(50) != 27:
        pass
    cv2.destroyAllWindows()
    WorldMap.save_pois(pois_path, prof.map_name or "UNKNOWN",
                       float(prof.world_size_m or 8000.0), pois)
    print(f"wrote {pois_path} with {len(pois)} places")


def preview(video: str, profile_path: str, out_png: str, at: float = 90.0,
            read: bool = True) -> Dict[str, Any]:
    frame, info = grab_frame(video, at)
    prof = MapProfile.load(profile_path).for_frame_size(info.width, info.height)
    world = WorldMap(tuple(int(v) for v in prof.map_area),  # type: ignore[arg-type]
                     float(prof.world_size_m or 8000.0), prof.map_name)
    canvas = frame.copy()
    x, y, w, h = [int(v) for v in prof.map_area]
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 255), 2)
    step = w / world.grid_cells
    for i in range(1, world.grid_cells):
        cv2.line(canvas, (int(x + i * step), y), (int(x + i * step), y + h), (0, 160, 160), 1)
        cv2.line(canvas, (x, int(y + i * step)), (x + w, int(y + i * step)), (0, 160, 160), 1)
    for gx in range(world.grid_cells):
        for gy in range(world.grid_cells):
            cv2.putText(canvas, world.grid_ref(gx * world.world_size_m / world.grid_cells + 1,
                                               gy * world.world_size_m / world.grid_cells + 1),
                        (int(x + gx * step + 4), int(y + gy * step + 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 160, 160), 1, cv2.LINE_AA)
    out: Dict[str, Any] = {"image": out_png, "resolution": [info.width, info.height],
                           "at": at, "map": prof.map_name,
                           "metres_per_pixel": round(world.metres_per_pixel(), 3),
                           "two_dots_merge_below_m": round(
                               2 * float(prof.dots.get("single_area", 60.0)) ** 0.5
                               * world.metres_per_pixel(), 1)}
    dots = detect_dots(frame, prof)
    if read and dots:
        read_dot_labels(frame, prof, dots, colours=ColourBook())
    lab = prof.label
    for d in dots:
        cv2.circle(canvas, (int(d.px), int(d.py)), 7, (0, 255, 0), 1)
        bx = int(d.px + float(lab["dx"]))
        by = int(d.py + float(lab["dy"]))
        cv2.rectangle(canvas, (bx, by), (bx + int(float(lab["w"])), by + int(float(lab["h"]))),
                      (255, 0, 255), 1)
        if d.team:
            cv2.putText(canvas, f"{d.team}:{d.player or '?'}", (int(d.px) + 8, int(d.py) + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    circles = read_circles(frame, prof, world)
    for name, c in circles.items():
        if c is None:
            continue
        px, py = world.to_pixels(c.cx, c.cy)
        cv2.circle(canvas, (int(px), int(py)), int(c.r / world.metres_per_pixel()),
                   (0, 0, 255), 1)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_png, canvas)
    out["dots_detected"] = len(dots)
    out["dots_clustered"] = sum(1 for d in dots if d.cluster_size > 1)
    out["labels_read"] = sum(1 for d in dots if d.team)
    out["sample_labels"] = [d.label_raw for d in dots[:8]]
    out["circles"] = {k: (None if v is None else
                          {"centre_m": [round(v.cx, 1), round(v.cy, 1)],
                           "radius_m": round(v.r, 1), "confidence": round(v.confidence, 3)})
                      for k, v in circles.items()}
    return out
