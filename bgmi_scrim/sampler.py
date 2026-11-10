"""Frame sampling: 2 fps baseline, 5 fps bursts around detected changes.

Decoding is sequential (``grab`` for skipped frames, ``retrieve`` for wanted
ones) because seeking in OBS-captured mp4 with variable keyframe spacing is
unreliable. Two passes are made over the file: a cheap pass that reads only
the lobby counters to find the moments where something changed, and a full
pass whose sample schedule is the 2 fps baseline plus a 5 fps burst inside a
window around every change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return VideoInfo(path, fps, n, w, h)


def base_schedule(info: VideoInfo, fps: float) -> List[int]:
    """Frame indices for a uniform sampling rate."""
    if info.fps <= 0 or fps <= 0:
        return []
    step = max(1, int(round(info.fps / fps)))
    return list(range(0, max(1, info.frame_count), step))


def burst_schedule(info: VideoInfo, change_times: Sequence[float],
                   burst_fps: float, window_s: float) -> List[int]:
    """Frame indices for 5 fps windows centred on each change time."""
    if not change_times:
        return []
    step = max(1, int(round(info.fps / burst_fps)))
    half = window_s / 2.0
    out: List[int] = []
    for t in change_times:
        f0 = max(0, int((t - half) * info.fps))
        f1 = min(info.frame_count - 1, int((t + half) * info.fps))
        out.extend(range(f0, f1 + 1, step))
    return out


def merge_schedules(*schedules: Iterable[int]) -> List[int]:
    s = set()
    for sch in schedules:
        s.update(int(i) for i in sch)
    return sorted(s)


def iter_frames(path: str, indices: Sequence[int]) -> Iterator[Tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_seconds, frame) for the requested indices."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    wanted = sorted(set(int(i) for i in indices))
    if not wanted:
        cap.release()
        return
    pos = 0
    wi = 0
    try:
        while wi < len(wanted):
            target = wanted[wi]
            while pos < target:
                if not cap.grab():
                    return
                pos += 1
            ok, frame = cap.read()
            if not ok:
                return
            yield pos, pos / fps, frame
            pos += 1
            wi += 1
            while wi < len(wanted) and wanted[wi] < pos:
                wi += 1
    finally:
        cap.release()
