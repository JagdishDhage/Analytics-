"""Synthetic observer-map footage with exact positional ground truth.

Test scaffolding for phase 3, the counterpart of ``synth.py``. It renders a
map viewport with terrain, named places, the two zone circles and one labelled
dot per living player, and writes the true world position of every player at
every sampled moment. Numbers measured on it are numbers on synthetic
footage.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .map_world import WorldMap, Poi

TEAMS = ["TX", "GB", "SOUL", "GODL", "ENT", "ORCA", "BLIND", "HYD", "REV", "SKY",
         "VLT", "NGE", "MARC", "AUT", "VST", "TSM", "XSP", "IQOO", "ZEUS", "NOVA",
         "FS", "GXR", "MDY", "K9", "OR"]
PLAYERS = ["ALFA", "BRAVO", "CHARL", "DELTA"]
POI_NAMES = ["PRISON", "SCHOOL", "MYLTA", "ROZHOK", "GATKA", "FARM", "QUARRY", "SHELTER",
             "NOVO", "LIPOVKA", "MANSION", "DOCKS", "RUINS", "HOSPITAL", "POWER",
             "KAMESHKI", "SEVERNY", "STALBER", "PRIMORSK", "GEORGO", "ZHARKI",
             "POCHINK", "YASNAYA", "MILBASE", "WATER"]


def _team_colour(i: int) -> Tuple[int, int, int]:
    hue = int((i * 180 / 25 + 7) % 180)
    hsv = np.uint8([[[hue, 230, 245]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


@dataclass
class MapSim:
    seed: int = 5
    duration: float = 420.0
    world_size_m: float = 8000.0
    map_name: str = "ERANGEL"
    our_team: str = "TX"
    n_teams: int = 25
    pois: List[Poi] = field(default_factory=list)
    phases: List[Dict[str, Any]] = field(default_factory=list)
    deaths: Dict[Tuple[str, str], float] = field(default_factory=dict)
    paths: Dict[Tuple[str, str], List[Tuple[float, float, float]]] = field(default_factory=dict)

    def build(self) -> "MapSim":
        rng = random.Random(self.seed)
        S = self.world_size_m
        # ---- places ----
        self.pois = []
        for i, name in enumerate(POI_NAMES[:self.n_teams]):
            gx, gy = i % 5, i // 5
            self.pois.append(Poi(name,
                                 S * (0.12 + 0.19 * gx) + rng.uniform(-150, 150),
                                 S * (0.12 + 0.19 * gy) + rng.uniform(-150, 150),
                                 220.0))
        # ---- zone phases: each circle inside the previous ----
        cx, cy, r = S / 2, S / 2, S * 0.45
        t = 30.0
        self.phases = [{"n": 1, "announced": 0.0, "cx": cx, "cy": cy, "r": r,
                        "closed": 25.0}]
        for n in range(2, 9):
            shrink = rng.uniform(0.52, 0.66)
            nr = r * shrink
            ang = rng.uniform(0, 2 * math.pi)
            off = (r - nr) * rng.uniform(0.0, 0.7)
            cx, cy = cx + off * math.cos(ang), cy + off * math.sin(ang)
            cx = min(max(nr, cx), S - nr)
            cy = min(max(nr, cy), S - nr)
            r = nr
            dur = max(28.0, (self.duration - 40) / 8)
            self.phases.append({"n": n, "announced": t, "cx": cx, "cy": cy, "r": r,
                                "closed": t + dur * 0.75})
            t += dur
            if t > self.duration - 20:
                break
        # ---- per-player paths ----
        plane_a = (rng.uniform(0, S), 0.0)
        plane_b = (rng.uniform(0, S), S)
        for ti in range(self.n_teams):
            team = TEAMS[ti]
            poi = self.pois[ti % len(self.pois)]
            drop = (poi.x_m + rng.uniform(-60, 60), poi.y_m + rng.uniform(-60, 60))
            jump_t = 2.0 + 10.0 * (ti / self.n_teams)
            air_from = (plane_a[0] + (plane_b[0] - plane_a[0]) * (ti / self.n_teams),
                        plane_a[1] + (plane_b[1] - plane_a[1]) * (ti / self.n_teams))
            land_t = jump_t + rng.uniform(14.0, 20.0)
            for pi, pname in enumerate(PLAYERS):
                key = (team, pname)
                off = (rng.uniform(-35, 35), rng.uniform(-35, 35))
                pts: List[Tuple[float, float, float]] = [(0.0, air_from[0], air_from[1]),
                                                         (jump_t, air_from[0], air_from[1])]
                pts.append((land_t, drop[0] + off[0], drop[1] + off[1]))
                # loot near the drop
                cur = (drop[0] + off[0], drop[1] + off[1])
                tt = land_t
                while tt < land_t + rng.uniform(45, 75):
                    tt += rng.uniform(6, 12)
                    cur = (cur[0] + rng.uniform(-45, 45), cur[1] + rng.uniform(-45, 45))
                    pts.append((tt, cur[0], cur[1]))
                # one rotation per phase, towards a point inside the next circle
                for ph in self.phases[1:]:
                    start = ph["announced"] + rng.uniform(5, 40)
                    ang = rng.uniform(0, 2 * math.pi)
                    rad = ph["r"] * rng.uniform(0.25, 0.9)
                    tgt = (ph["cx"] + rad * math.cos(ang) + off[0],
                           ph["cy"] + rad * math.sin(ang) + off[1])
                    speed = rng.choice([4.5, 4.5, 16.0])     # foot or vehicle
                    dist = math.dist(cur, tgt)
                    travel = dist / speed
                    if start < tt:
                        start = tt + 1.0
                    pts.append((start, cur[0], cur[1]))
                    pts.append((start + travel, tgt[0], tgt[1]))
                    cur = tgt
                    tt = start + travel
                    # settle and loot a little
                    for _ in range(2):
                        tt += rng.uniform(8, 16)
                        cur = (cur[0] + rng.uniform(-25, 25), cur[1] + rng.uniform(-25, 25))
                        pts.append((tt, cur[0], cur[1]))
                pts.sort(key=lambda p: p[0])
                self.paths[key] = pts
        # ---- deaths: teams die off roughly in drop order, ours mid-late ----
        order = [t for t in TEAMS[:self.n_teams] if t != self.our_team]
        rng.shuffle(order)
        order = order[: self.n_teams - 2] + [self.our_team] + order[self.n_teams - 2:]
        for i, team in enumerate(order):
            base = 60.0 + (self.duration - 90.0) * (i / max(1, len(order)))
            for pi, pname in enumerate(PLAYERS):
                if i >= len(order) - 1:
                    continue      # last team survives
                self.deaths[(team, pname)] = min(self.duration - 2,
                                                 base + rng.uniform(0, 25))
        return self

    # ---------------- state ----------------
    def position(self, team: str, player: str, t: float) -> Optional[Tuple[float, float]]:
        d = self.deaths.get((team, player))
        if d is not None and t >= d:
            return None
        pts = self.paths.get((team, player))
        if not pts:
            return None
        if t <= pts[0][0]:
            return pts[0][1], pts[0][2]
        for a, b in zip(pts, pts[1:]):
            if a[0] <= t <= b[0]:
                span = max(1e-6, b[0] - a[0])
                f = (t - a[0]) / span
                return a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f
        return pts[-1][1], pts[-1][2]

    def zone_at(self, t: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        """(safe circle, blue circle) at time t, in world metres."""
        cur = self.phases[0]
        nxt = None
        for ph in self.phases:
            if ph["announced"] <= t:
                cur = ph
            elif nxt is None:
                nxt = ph
        safe = {"cx": cur["cx"], "cy": cur["cy"], "r": cur["r"]}
        prev = self.phases[max(0, cur["n"] - 2)]
        if t < cur["closed"] and cur["n"] > 1:
            span = max(1e-6, cur["closed"] - cur["announced"])
            f = min(1.0, max(0.0, (t - cur["announced"]) / span))
            blue = {"cx": prev["cx"] + (cur["cx"] - prev["cx"]) * f,
                    "cy": prev["cy"] + (cur["cy"] - prev["cy"]) * f,
                    "r": prev["r"] + (cur["r"] - prev["r"]) * f}
        else:
            blue = dict(safe)
        return safe, blue


# ---------------- rendering ----------------
def _terrain(w: int, h: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.normal(0.5, 0.18, (24, 24))
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (0, 0), 9)
    base = np.zeros((h, w, 3), np.float32)
    base[..., 0] = 70 + 40 * big          # muted blue-green terrain, low saturation
    base[..., 1] = 88 + 46 * big
    base[..., 2] = 74 + 38 * big
    img = np.clip(base, 0, 255).astype(np.uint8)
    for i in range(6):                     # roads
        p1 = (int(rng.integers(0, w)), 0)
        p2 = (int(rng.integers(0, w)), h)
        cv2.line(img, p1, p2, (150, 150, 150), 2, cv2.LINE_AA)
    return img


def render_map_frame(sim: MapSim, t: float, width: int, height: int,
                     map_area: Tuple[int, int, int, int], noise: float = 0.0,
                     label_scale: float = 0.38) -> np.ndarray:
    x0, y0, mw, mh = map_area
    frame = np.full((height, width, 3), 22, np.uint8)
    terrain = _terrain(mw, mh, sim.seed)
    frame[y0:y0 + mh, x0:x0 + mw] = terrain
    world = WorldMap(map_area, sim.world_size_m, sim.map_name, sim.pois)

    for p in sim.pois:                      # place names, in white like the HUD
        px, py = world.to_pixels(p.x_m, p.y_m)
        cv2.putText(frame, p.name, (int(px) - 26, int(py)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (215, 215, 215), 1, cv2.LINE_AA)

    safe, blue = sim.zone_at(t)
    bpx, bpy = world.to_pixels(blue["cx"], blue["cy"])
    cv2.circle(frame, (int(bpx), int(bpy)), int(blue["r"] / world.metres_per_pixel()),
               (235, 150, 40), 3, cv2.LINE_AA)
    spx, spy = world.to_pixels(safe["cx"], safe["cy"])
    cv2.circle(frame, (int(spx), int(spy)), int(safe["r"] / world.metres_per_pixel()),
               (255, 255, 255), 2, cv2.LINE_AA)

    for ti in range(sim.n_teams):
        team = TEAMS[ti]
        colour = _team_colour(ti)
        for pname in PLAYERS:
            pos = sim.position(team, pname, t)
            if pos is None:
                continue
            px, py = world.to_pixels(pos[0], pos[1])
            if not (x0 <= px < x0 + mw and y0 <= py < y0 + mh):
                continue
            cv2.circle(frame, (int(px), int(py)), 5, colour, -1, cv2.LINE_AA)
            cv2.putText(frame, f"{team}x{pname}", (int(px) + 7, int(py) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, label_scale, (245, 245, 245), 1,
                        cv2.LINE_AA)
    if noise > 0:
        n = np.random.default_rng(int(t * 1000) % 99991).normal(0, noise * 255, frame.shape)
        frame = np.clip(frame.astype(np.int16) + n.astype(np.int16), 0, 255).astype(np.uint8)
    return frame


def labels_for(sim: MapSim, label_fps: float = 2.0) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = [{"kind": "map_info", "map": sim.map_name,
                                  "world_size_m": sim.world_size_m,
                                  "our_team": sim.our_team}]
    for p in sim.pois:
        out.append({"kind": "poi", "name": p.name, "x_m": round(p.x_m, 1),
                    "y_m": round(p.y_m, 1), "radius_m": p.radius_m})
    step = 1.0 / label_fps
    t = 0.0
    while t < sim.duration:
        safe, blue = sim.zone_at(t)
        for ring, c in (("safe", safe), ("blue", blue)):
            out.append({"kind": "zone", "t": round(t, 3), "ring": ring,
                        "cx_m": round(c["cx"], 1), "cy_m": round(c["cy"], 1),
                        "r_m": round(c["r"], 1)})
        for ti in range(sim.n_teams):
            team = TEAMS[ti]
            for pname in PLAYERS:
                pos = sim.position(team, pname, t)
                if pos is None:
                    continue
                out.append({"kind": "pos", "t": round(t, 3), "team": team, "player": pname,
                            "x_m": round(pos[0], 1), "y_m": round(pos[1], 1)})
        t += step
    for (team, player), dt in sorted(sim.deaths.items()):
        out.append({"kind": "death", "t": round(dt, 2), "team": team, "player": player})
    return out


def write_map_clip(out_video: str, out_labels: str, width: int = 1920, height: int = 1080,
                   map_area: Optional[Sequence[int]] = None, fps: float = 30.0,
                   duration: float = 420.0, seed: int = 5, noise: float = 0.0,
                   our_team: str = "TX", map_name: str = "ERANGEL",
                   world_size_m: float = 8000.0, n_teams: int = 25,
                   sample_fps: Optional[float] = None) -> Dict[str, Any]:
    """Render the clip. ``sample_fps`` renders only the frames an analyzer will
    sample (a much smaller file) while keeping real timestamps."""
    area = tuple(map_area) if map_area else (int(width * 0.22), int(height * 0.04),
                                             int(height * 0.9), int(height * 0.9))
    sim = MapSim(seed=seed, duration=duration, world_size_m=world_size_m,
                 map_name=map_name, our_team=our_team, n_teams=n_teams).build()
    Path(out_video).parent.mkdir(parents=True, exist_ok=True)
    out_fps = float(sample_fps or fps)
    vw = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height))
    if not vw.isOpened():
        raise IOError("cannot open VideoWriter")
    n = int(duration * out_fps)
    for i in range(n):
        vw.write(render_map_frame(sim, i / out_fps, width, height, area, noise=noise))
    vw.release()
    Path(out_labels).write_text(
        "\n".join(json.dumps(x, sort_keys=True) for x in labels_for(sim)) + "\n")
    return {"video": out_video, "labels": out_labels, "map_area": list(area),
            "teams": TEAMS[:n_teams], "our_team": our_team,
            "phases": len(sim.phases), "fps": out_fps,
            "pois": [p.name for p in sim.pois]}
