"""Calibration: write and verify a profile for a given video.

``calibrate`` has three modes, because the machine that records scrims is not
always the machine with a display:

* interactive  -- OpenCV ROI picker, one box per HUD field (needs highgui)
* template     -- scale an existing profile to this video's resolution
* preview      -- draw the profile's boxes on a frame and print what each ROI
                  currently reads, which is how a profile gets verified

Nothing here is used at analysis time; profiles are data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import hud, sampler
from .profile import Profile

# Fields a usable profile needs, with the kind of box each one is.
REQUIRED = [
    ("teams_alive", "numeral", "lobby 'teams alive' counter digits"),
    ("players_alive", "numeral", "lobby 'players alive' counter digits"),
    ("zone_phase", "numeral", "zone/phase number"),
    ("zone_state", "colour_state", "blue-zone indicator (shrinking vs waiting)"),
    ("map_name", "text", "map name text"),
    ("kill_feed", "feed", "TOP kill-feed row only (rows/row_pitch set after)"),
    ("team_panel", "team_panel", "team list panel, top-left corner at row 1"),
    ("placement", "numeral", "rank number on the elimination card"),
    ("winner_banner", "text", "winning team tag on the end-of-match banner"),
]


def grab_frame(video: str, at: float = 60.0) -> Tuple[np.ndarray, sampler.VideoInfo]:
    info = sampler.probe(video)
    idx = min(max(0, int(at * info.fps)), max(0, info.frame_count - 1))
    for _, _, frame in sampler.iter_frames(video, [idx]):
        return frame, info
    raise IOError(f"could not read a frame from {video}")


def _gui_available() -> bool:
    try:
        cv2.namedWindow("probe", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("probe")
        return True
    except Exception:
        return False


def interactive(video: str, out: str, base: Optional[str] = None, at: float = 60.0,
                our_team: Optional[str] = None) -> Profile:
    frame, info = grab_frame(video, at)
    prof = Profile.load(base).for_frame_size(info.width, info.height) if base else Profile(
        name=Path(out).stem, resolution=(info.width, info.height))
    prof.resolution = (info.width, info.height)
    prof.name = Path(out).stem
    if our_team:
        prof.our_team = our_team
    if not _gui_available():
        raise RuntimeError(
            "OpenCV highgui is not available here. Use 'calibrate --template' to scale an "
            "existing profile to this resolution, edit the boxes in the JSON, then check "
            "them with 'calibrate --preview'.")
    print("Drag a box, then ENTER to accept or 'c' to skip. ESC aborts.")
    for name, kind, desc in REQUIRED:
        title = f"{name} -- {desc}"
        box = cv2.selectROI(title, frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(title)
        if box[2] <= 0 or box[3] <= 0:
            print(f"  {name}: skipped")
            continue
        meta: Dict[str, Any] = dict(prof.rois[name].meta) if name in prof.rois else {}
        from .profile import Roi
        prof.rois[name] = Roi(name, tuple(int(v) for v in box), kind, meta)
        print(f"  {name}: {tuple(int(v) for v in box)}")
    prof.save(out)
    print(f"wrote {out}")
    print("Panel/feed row geometry (rows, row_pitch, slot_x0, slot_pitch, slot_w, slot_h) "
          "and the HSV bands are edited in the JSON, then verified with --preview.")
    return prof


def template(video: str, out: str, base: str, our_team: Optional[str] = None) -> Profile:
    info = sampler.probe(video)
    prof = Profile.load(base).for_frame_size(info.width, info.height)
    prof.name = Path(out).stem
    prof.resolution = (info.width, info.height)
    if our_team:
        prof.our_team = our_team
    prof.save(out)
    print(f"wrote {out} ({info.width}x{info.height}, scaled from {base})")
    return prof


def preview(video: str, profile_path: str, out_png: str, at: float = 60.0,
            read: bool = True) -> Dict[str, Any]:
    frame, info = grab_frame(video, at)
    prof = Profile.load(profile_path).for_frame_size(info.width, info.height)
    canvas = frame.copy()
    for name, roi in prof.rois.items():
        x, y, w, h = roi.box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(canvas, name, (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)
    panel = prof.rois.get("team_panel")
    if panel:
        rows = int(panel.meta.get("rows", 25))
        slots = int(panel.meta.get("slots", 4))
        for r in range(rows):
            for s in range(slots):
                x, y, w, h = hud._slot_box(panel, r, s)
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 200, 0), 1)
            x, y, w, h = hud._tag_box(panel, r)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 160, 0), 1)
    feed = prof.rois.get("kill_feed")
    if feed and feed.meta.get("rows"):
        x, y, w, _ = feed.box
        pitch = float(feed.meta["row_pitch"])
        h = int(round(float(feed.meta.get("row_h", pitch))))
        for r in range(int(feed.meta["rows"])):
            ry = y + int(round(r * pitch))
            cv2.rectangle(canvas, (x, ry), (x + w, ry + h), (255, 0, 255), 1)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_png, canvas)
    out: Dict[str, Any] = {"image": out_png, "resolution": [info.width, info.height],
                           "at": at}
    if read:
        out["readings"] = {
            "teams_alive": hud.read_counter(frame, prof, "teams_alive"),
            "players_alive": hud.read_counter(frame, prof, "players_alive"),
            "zone_phase": hud.read_zone_phase(frame, prof),
            "zone_state": hud.read_zone_state(frame, prof),
            "map_name": hud.read_choice_text(frame, prof, "map_name", prof.map_names),
            "kill_feed": hud.read_feed_lines(frame, prof),
            "placement": hud.read_placement(frame, prof),
            "winner_banner": hud.read_winner_tag(frame, prof),
        }
        pan = hud.read_team_panel(frame, prof, read_tags=True)
        out["readings"]["team_tags"] = {r: v for r, v in sorted(pan.tags.items())}
        out["readings"]["slot_states"] = {
            f"row{r+1}": [next((s.state for s in pan.slots if s.team_row == r and s.slot == i), None)
                          for i in range(int(panel.meta.get("slots", 4)) if panel else 0)]
            for r in range(min(5, pan.rows))
        }
        unreadable = [k for k, v in out["readings"].items()
                      if isinstance(v, tuple) and (v[0] is None or v[1] < prof.confidence_threshold)]
        out["low_confidence_rois"] = unreadable
    return out
