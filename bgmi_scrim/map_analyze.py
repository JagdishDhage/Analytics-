"""Phase 3 pipeline: map stream in, position tracks and map events out."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import sampler
from .map_dots import ColourBook, Dot, assign_teams_by_colour, detect_dots, read_dot_labels
from .map_movement import (DEFAULTS, Phase, Sample, Segment, ZoneSample,
                           edge_fraction, landing, phases, rotation_start_after_announce,
                           segments, squad_spread, time_outside, zone_entry)
from .map_profile import MapProfile
from .map_track import Detection, Track, Tracker
from .map_world import WorldMap, spread_m
from .map_zones import ZoneStabiliser, read_circles


@dataclass
class MapRunStats:
    samples: int = 0
    label_frames: int = 0
    labels_read: int = 0
    zone_frames: int = 0
    dots_seen: int = 0
    tracks_created: int = 0
    identified_samples: int = 0
    anonymous_samples: int = 0
    identity_conflicts: int = 0
    zone_fits_rejected: int = 0
    seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _world_for(prof: MapProfile) -> WorldMap:
    size = prof.world_size_m or 8000.0
    pois = []
    if prof.pois_file and Path(prof.pois_file).exists():
        pois, _, ws = WorldMap.load_pois(prof.pois_file)
        if ws:
            size = float(ws)
    return WorldMap(tuple(int(v) for v in prof.map_area), float(size),  # type: ignore[arg-type]
                    prof.map_name, pois)


class MapAnalyzer:
    def __init__(self, prof: MapProfile, our_team: Optional[str] = None,
                 workers: int = 4, verbose: bool = False):
        self.prof = prof
        self.our_team = our_team
        self.workers = workers
        self.verbose = verbose
        self.records: List[Dict[str, Any]] = []
        self.stats = MapRunStats()

    def run(self, path: str) -> Tuple[List[Dict[str, Any]], MapRunStats]:
        t0 = time.time()
        info = sampler.probe(path)
        prof = self.prof.for_frame_size(info.width, info.height)
        world = _world_for(prof)
        if not prof.calibrated:
            print(f"[mapanalyze] WARNING: profile '{prof.name}' is a template "
                  f"(_calibrated=false); its map box and colour bands are placeholders.")
        if self.verbose:
            print(f"[mapanalyze] {info.width}x{info.height} {info.fps:.2f}fps "
                  f"{info.duration:.1f}s, map {prof.map_name} "
                  f"{world.world_size_m:.0f} m, {world.metres_per_pixel():.2f} m/px")

        base_fps = float(prof.sampling.get("base_fps", 2.0))
        schedule = sampler.base_schedule(info, base_fps)
        label_every = max(1, int(round(float(prof.sampling.get("label_refresh_s", 10.0)) * base_fps)))
        zone_every = max(1, int(round(float(prof.sampling.get("zone_refresh_s", 2.0)) * base_fps)))
        prepass = int(prof.sampling.get("label_prepass_frames", 4))
        label_min_gap = float(prof.sampling.get("label_min_gap_s", 3.0))
        gate_m = float(prof.tracking.get("gate_floor_m", 60.0))
        last_label_t = -1e9

        colours = ColourBook(tol_deg=float(prof.dots.get("colour_tol_deg", 12.0)))
        stabiliser = ZoneStabiliser()
        pending: List[Dict[str, Any]] = []
        track_colour: Dict[int, List[Tuple[int, int, int]]] = {}
        tracker = Tracker(float(prof.tracking.get("max_speed_mps", 140.0)),
                          float(prof.tracking.get("max_miss_s", 8.0)),
                          float(prof.tracking.get("gate_floor_m", 60.0)))
        self.records.append({"type": "map_info", "map": prof.map_name,
                             "world_size_m": world.world_size_m,
                             "metres_per_pixel": round(world.metres_per_pixel(), 4),
                             "profile": prof.name, "our_team": self.our_team,
                             "pois": len(world.pois)})

        for i, (fi, t, frame) in enumerate(sampler.iter_frames(path, schedule)):
            self.stats.samples += 1
            dots = detect_dots(frame, prof)
            self.stats.dots_seen += len(dots)
            want_labels = (i < prepass) or (i % label_every == 0)
            if not want_labels and tracker.tracks and (t - last_label_t) >= label_min_gap:
                unknown = sum(1 for tr in tracker.tracks if tracker.needs_label(tr))
                want_labels = unknown > max(2, 0.25 * len(tracker.tracks))
            if want_labels and dots:
                # only read names next to dots whose track still needs one
                todo = [d for d in dots if self._needs_label_near(tracker, world, d, gate_m)]
                if todo:
                    read_dot_labels(frame, prof, todo, workers=self.workers, colours=colours)
                    self.stats.label_frames += 1
                    self.stats.labels_read += len(todo)
                last_label_t = t
            assign_teams_by_colour(dots, colours)

            dets = []
            for d in dots:
                wx, wy = world.to_world(d.px, d.py)
                dets.append(Detection(wx, wy, d.cluster_size, d.confidence,
                                      d.team, d.player, d.label_conf,
                                      team_conf=d.team_conf, colour=d.colour))
            before = tracker._next_id
            live = tracker.update(t, dets)
            self.stats.tracks_created += tracker._next_id - before
            for tr in live:
                pending.append({"t": round(t, 3), "track": tr.id,
                                "x_m": round(tr.x, 1), "y_m": round(tr.y, 1),
                                "cluster": tr.cluster,
                                "dot_confidence": round(min(tr.confidence, 1.0), 4)})
                track_colour.setdefault(tr.id, []).append(tr.colour)

            if i % zone_every == 0:
                for ring, raw in read_circles(frame, prof, world).items():
                    c = stabiliser.accept(ring, raw) if (
                        raw is not None and raw.confidence >= prof.confidence_threshold) else None
                    if c is None:
                        self.records.append({"type": "zone", "t": round(t, 3), "ring": ring,
                                             "state": "unknown", "confidence":
                                             round(raw.confidence, 3) if raw else 0.0})
                        continue
                    self.records.append({"type": "zone", "t": round(t, 3), "ring": ring,
                                         "cx_m": round(c.cx, 1), "cy_m": round(c.cy, 1),
                                         "r_m": round(c.r, 1),
                                         "confidence": round(c.confidence, 4)})
                self.stats.zone_frames += 1

        # ---- identity is settled once, with everything the run learned ----
        ident = self._settle_identities(tracker, colours, track_colour)
        for rec in pending:
            team, player, iconf, src = ident.get(rec["track"], (None, None, 0.0, ""))
            out = dict(rec)
            conf = out.pop("dot_confidence")
            out["type"] = "sample"
            out["place"] = world.place(out["x_m"], out["y_m"])
            out["confidence"] = round(conf * (0.5 + 0.5 * iconf), 4)
            if team is not None:
                out["team"] = team
                out["identity_confidence"] = round(iconf, 4)
                out["identity_source"] = src
                self.stats.identified_samples += 1
                if player:
                    out["player"] = player
            else:
                out["state"] = "unknown_identity"
                self.stats.anonymous_samples += 1
            self.records.append(out)
        self.records.sort(key=lambda r: (float(r.get("t", -1.0)), r.get("track", 0)))
        self.stats.zone_fits_rejected = stabiliser.rejected
        self.stats.identity_conflicts = tracker.id_switch_guard
        self.stats.seconds = time.time() - t0
        return self.records, self.stats

    @staticmethod
    def _needs_label_near(tracker: Tracker, world: WorldMap, dot: Dot,
                          gate_m: float) -> bool:
        """Is there a track at this dot that already knows who it is?"""
        wx, wy = world.to_world(dot.px, dot.py)
        best, best_d = None, gate_m
        for tr in tracker.tracks:
            d = ((tr.x - wx) ** 2 + (tr.y - wy) ** 2) ** 0.5
            if d < best_d:
                best, best_d = tr, d
        if best is None:
            return True
        team, player, conf = best.identity()
        return team is None or player is None or conf < 0.7

    def _settle_identities(self, tracker: Tracker, colours: ColourBook,
                           track_colour: Dict[int, List[Tuple[int, int, int]]]
                           ) -> Dict[int, Tuple[Optional[str], Optional[str], float, str]]:
        """Give every track the identity it earned over the whole run.

        A squad stacked on its drop is one dot with four names piled on top of
        each other, and nothing readable comes off it. The same dot is
        readable two minutes later when the squad spreads, and the colour book
        is fuller still by the end of the match -- so identity is decided once,
        at the end, and applied to the track's whole history rather than to
        the moment the name happened to be legible.
        """
        out: Dict[int, Tuple[Optional[str], Optional[str], float, str]] = {}
        min_vote = float(self.prof.tracking.get("min_identity_vote", 1.2))
        min_conf = float(self.prof.tracking.get("min_identity_conf", 0.65))
        for tr in tracker.all_tracks():
            team, player, conf = tr.identity()
            src = "label" if team is not None else ""
            if team is not None and (sum(tr.team_votes.values()) < min_vote
                                     or conf < min_conf):
                # votes split between two teams is not an identification
                team, player, conf, src = None, None, 0.0, ""
            if team is not None:
                cols = track_colour.get(tr.id, [])
                if cols and colours.agrees(team, cols[len(cols) // 2]) is False:
                    # the dot's own colour says this name belongs to someone else
                    team, player, conf, src = None, None, 0.0, ""
            if team is None:
                cols = track_colour.get(tr.id, [])
                if cols:
                    mid = cols[len(cols) // 2]
                    team, cconf = colours.team_from_colour(mid)
                    if team is not None:
                        conf, src = cconf * 0.9, "colour"
            out[tr.id] = (team, player, conf, src)
        return out


# ---------------- derivation ----------------
def _zone_samples(records: Sequence[Dict[str, Any]], threshold: float) -> List[ZoneSample]:
    out = []
    for r in records:
        if r.get("type") != "zone" or r.get("state") == "unknown":
            continue
        if float(r.get("confidence", 0)) < threshold:
            continue
        out.append(ZoneSample(float(r["t"]), r["ring"], float(r["cx_m"]),
                              float(r["cy_m"]), float(r["r_m"]),
                              float(r.get("confidence", 1.0))))
    out.sort(key=lambda z: z.t)
    return out


def _player_paths(records: Sequence[Dict[str, Any]], threshold: float
                  ) -> Dict[Tuple[str, str], List[Sample]]:
    """Paths of players whose own name was read."""
    out: Dict[Tuple[str, str], List[Sample]] = {}
    for r in records:
        if r.get("type") != "sample" or not r.get("player") or not r.get("team"):
            continue
        if float(r.get("confidence", 0)) < threshold:
            continue
        out.setdefault((r["team"], r["player"]), []).append(
            Sample(float(r["t"]), float(r["x_m"]), float(r["y_m"]),
                   float(r.get("confidence", 1.0)), int(r.get("cluster", 1))))
    for v in out.values():
        v.sort(key=lambda s: s.t)
    return out


def _team_paths(records: Sequence[Dict[str, Any]], threshold: float
                ) -> Dict[str, List[Sample]]:
    """One path per team: the median position of that team's dots at each sample.

    This is the unit the medium actually supports. Four team-mates in a
    building are one dot at eight metres per pixel, so asking where a named
    player was is often unanswerable, while asking where the team was is not.
    The median rather than the mean keeps one misidentified dot from dragging
    the whole squad across the map.
    """
    by_t: Dict[str, Dict[float, List[Tuple[float, float, int, float]]]] = {}
    for r in records:
        if r.get("type") != "sample" or not r.get("team"):
            continue
        if float(r.get("confidence", 0)) < threshold:
            continue
        by_t.setdefault(r["team"], {}).setdefault(round(float(r["t"]), 3), []).append(
            (float(r["x_m"]), float(r["y_m"]), int(r.get("cluster", 1)),
             float(r.get("confidence", 1.0))))
    out: Dict[str, List[Sample]] = {}
    link_m = 350.0
    max_speed = 60.0          # a squad on the ground, with headroom for vehicles
    drop_window_s = 150.0
    for team, times in by_t.items():
        path: List[Sample] = []
        prev: Optional[Tuple[float, float]] = None
        prev_t: Optional[float] = None
        t_first = min(times) if times else 0.0
        for t in sorted(times):
            rows = times[t]
            groups = _spatial_groups(rows, link_m)
            if prev is None:
                pick = max(groups, key=lambda g: (sum(r[2] for r in g), len(g)))
            else:
                pick = min(groups, key=lambda g: min(
                    math.dist((r[0], r[1]), prev) for r in g))
            cx = sum(r[0] for r in pick) / len(pick)
            cy = sum(r[1] for r in pick) / len(pick)
            if prev is not None and prev_t is not None:
                dt = max(1e-3, t - prev_t)
                # during the drop a squad really does cross the map, under a
                # parachute; after it, nothing does
                cap = 400.0 if t <= t_first + drop_window_s else max_speed
                if math.dist((cx, cy), prev) / dt > cap:
                    # No squad crosses the map in half a second. The median of
                    # a team's dots flips between separated players, which
                    # would otherwise read as a rotation at aircraft speed:
                    # leave a hole instead of inventing the journey.
                    continue
            path.append(Sample(t, cx, cy, sum(r[3] for r in pick) / len(pick),
                               sum(r[2] for r in pick)))
            prev = (cx, cy)
            prev_t = t
        out[team] = path
    return out


def _spatial_groups(rows: Sequence[Tuple[float, float, int, float]], link_m: float
                    ) -> List[List[Tuple[float, float, int, float]]]:
    """Single-link grouping, so one stray dot cannot move a whole squad."""
    groups: List[List[Tuple[float, float, int, float]]] = []
    for r in rows:
        hit = None
        for g in groups:
            if any(math.dist((r[0], r[1]), (o[0], o[1])) <= link_m for o in g):
                hit = g
                break
        if hit is None:
            groups.append([r])
        else:
            hit.append(r)
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if any(math.dist((a[0], a[1]), (b[0], b[1])) <= link_m
                       for a in groups[i] for b in groups[j]):
                    groups[i].extend(groups[j])
                    del groups[j]
                    merged = True
                    break
            if merged:
                break
    return groups


def _unit_events(who: Dict[str, Any], samples: Sequence[Sample], world: WorldMap,
                 ph_list: Sequence[Phase], blue: Sequence[ZoneSample],
                 cfg: Dict[str, Any], t0: Optional[float] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    conf = sum(s.confidence for s in samples) / len(samples)
    land, saw_descent = landing(samples, cfg, t0)
    if land is not None:
        rec: Dict[str, Any] = {"type": "landing", "t": round(land.t, 2), **who,
                               "x_m": round(land.x, 1), "y_m": round(land.y, 1),
                               "place": world.place(land.x, land.y),
                               "confidence": round(conf if saw_descent else conf * 0.6, 3)}
        if not saw_descent:
            rec["state"] = "no_descent_seen"
            rec["note"] = "stream starts after the drop; this is the first tracked position"
        out.append(rec)
    ground = [s for s in samples if land is None or s.t >= land.t]
    for seg in segments(ground, cfg):
        rec: Dict[str, Any] = {"type": seg.kind, "t": round(seg.t0, 2),
                               "t_end": round(seg.t1, 2), **who,
                               "duration_s": round(seg.duration, 1),
                               "from": world.place(seg.x0, seg.y0),
                               "to": world.place(seg.x1, seg.y1),
                               "distance_m": round(seg.distance_m, 1),
                               "displacement_m": round(seg.displacement_m, 1),
                               "mean_speed_mps": round(seg.mean_speed, 2),
                               "confidence": round(conf, 3)}
        if seg.kind == "rotation":
            rec["mode"] = seg.mode
            ph = None
            for p in ph_list:
                if p.announced_t <= seg.t0:
                    ph = p
            if ph is not None:
                rec["phase"] = ph.number
                rec["after_announce_s"] = round(seg.t0 - ph.announced_t, 1)
                if ph.closed_t is not None:
                    margin = ph.closed_t - seg.t1
                    rec["before_close_s"] = round(margin, 1)
                    if margin < 0:
                        rec["arrived_after_close_s"] = round(-margin, 1)
        out.append(rec)
    for ph in ph_list:
        hit = zone_entry(samples, ph)
        if hit is None:
            out.append({"type": "zone_entry", "t": round(ph.announced_t, 2), **who,
                        "phase": ph.number, "state": "never_entered",
                        "confidence": round(conf, 3)})
            continue
        t_in, margin = hit
        rec = {"type": "zone_entry", "t": round(t_in, 2), **who, "phase": ph.number,
               "confidence": round(conf, 3)}
        if margin == margin:
            rec["margin_s"] = round(margin, 1)
        out.append(rec)
    outside = time_outside(samples, ph_list, blue)
    ef = edge_fraction(samples, ph_list, float(cfg.get("edge_band", 0.25)))
    out.append({"type": "zone_summary", "t": round(samples[-1].t, 2), **who,
                "time_outside_blue_s": round(outside, 1),
                "edge_time_fraction": round(ef, 3) if ef is not None else None,
                "tracked_from_s": round(samples[0].t, 2),
                "tracked_to_s": round(samples[-1].t, 2),
                "confidence": round(conf, 3)})
    return out


def derive_events(records: Sequence[Dict[str, Any]], prof: MapProfile,
                  world: Optional[WorldMap] = None,
                  cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Landings, loot stretches, rotations, phases and zone entries.

    Everything is derived twice: once per team, from the team's median dot,
    and once per named player where the stream gave a name.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    thr = prof.confidence_threshold
    world = world or _world_for(prof)
    zs = _zone_samples(records, thr)
    ph_list = phases(zs)
    blue = [z for z in zs if z.ring == "blue"]
    out: List[Dict[str, Any]] = []

    for ph in ph_list:
        out.append({"type": "zone_phase", "t": round(ph.announced_t, 2), "phase": ph.number,
                    "cx_m": round(ph.cx, 1), "cy_m": round(ph.cy, 1), "r_m": round(ph.r, 1),
                    "place": world.place(ph.cx, ph.cy),
                    "closed_t": round(ph.closed_t, 2) if ph.closed_t is not None else None})

    teams = _team_paths(records, thr)
    t0 = min((float(r["t"]) for r in records if r.get("type") == "sample"), default=0.0)
    for team, samples in sorted(teams.items()):
        if len(samples) < 2:
            continue
        members = max(s.cluster for s in samples)
        out.extend(_unit_events({"team": team, "unit": "team", "dots_max": members},
                                samples, world, ph_list, blue, cfg, t0))
    for (team, player), samples in sorted(_player_paths(records, thr).items()):
        if len(samples) < 2:
            continue
        out.extend(_unit_events({"team": team, "player": player, "unit": "player"},
                                samples, world, ph_list, blue, cfg, t0))

    for team, samples in teams.items():
        players = _player_paths(records, thr)
        mates = {p: ss for (tm, p), ss in players.items() if tm == team}
        if len(mates) >= 2:
            for s in samples[::10]:
                sp = squad_spread(mates, s.t)
                if sp is not None:
                    out.append({"type": "squad_spread", "t": round(s.t, 1), "team": team,
                                "value_m": round(sp, 1), "players_seen": len(mates)})
    out.sort(key=lambda r: (r["t"], r["type"], r.get("team", "")))
    return out


def analyze_map_video(video: str, profile: MapProfile, tracks_out: Optional[str] = None,
                      events_out: Optional[str] = None, our_team: Optional[str] = None,
                      workers: int = 4, verbose: bool = False
                      ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], MapRunStats]:
    an = MapAnalyzer(profile, our_team=our_team, workers=workers, verbose=verbose)
    records, stats = an.run(video)
    events = derive_events(records, profile)
    if tracks_out:
        Path(tracks_out).parent.mkdir(parents=True, exist_ok=True)
        Path(tracks_out).write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n")
    if events_out:
        Path(events_out).parent.mkdir(parents=True, exist_ok=True)
        Path(events_out).write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in events) + "\n")
    return records, events, stats
