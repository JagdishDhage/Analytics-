"""Phase-1 pipeline: video in, events.jsonl out."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import hud, sampler
from .events import (Event, KILL, KNOCK, MAP_NAME, MATCH_END, MATCH_START,
                     PLACEMENT, PLAYERS_ALIVE, TEAM_ALIVE_COUNT, TEAMS_ALIVE,
                     WINNER, ZONE_PHASE, ZONE_STATE)
from .fuse import Fuser, PanelEvent
from .killfeed import FeedTracker
from .profile import Profile
from .statemachine import UNKNOWN, ConfirmedField


class PanelTagResolver:
    """Resolve which team sits on which panel row.

    Team identity is static for most of a match, so it is resolved once in a
    short pre-pass over frames spread across the video, by majority vote with
    a roster-constrained unique assignment (two rows cannot be the same team).
    Later re-reads can move a row if they disagree decisively on two
    consecutive epochs, which is how a panel that re-sorts as teams die is
    picked up.
    """

    def __init__(self, rows: int, roster: Optional[List[str]] = None):
        self.rows = rows
        self.roster = list(roster or [])
        self.votes: Dict[int, Dict[str, float]] = {}
        self.assigned: Dict[int, str] = {}
        self.epoch_votes: Dict[int, Dict[str, float]] = {}
        self.pending_change: Dict[int, Tuple[str, int]] = {}
        self.reorders = 0

    def vote(self, row: int, tag: str, conf: float) -> None:
        self.votes.setdefault(row, {})
        self.votes[row][tag] = self.votes[row].get(tag, 0.0) + max(conf, 0.01)
        self.epoch_votes.setdefault(row, {})
        self.epoch_votes[row][tag] = self.epoch_votes[row].get(tag, 0.0) + max(conf, 0.01)

    def resolve(self) -> Dict[int, str]:
        """Greedy unique assignment of roster tags to rows by vote weight."""
        pairs: List[Tuple[float, int, str]] = []
        for r, v in self.votes.items():
            for tag, w in v.items():
                pairs.append((w, r, tag))
        pairs.sort(reverse=True)
        used_rows: set = set()
        used_tags: set = set()
        out: Dict[int, str] = {}
        for w, r, tag in pairs:
            if r in used_rows:
                continue
            if self.roster and tag in used_tags:
                continue
            out[r] = tag
            used_rows.add(r)
            used_tags.add(tag)
        self.assigned = out
        self.epoch_votes = {}
        return out

    def end_epoch(self) -> None:
        """Apply a decisive, repeated disagreement as a row reassignment."""
        for r, v in self.epoch_votes.items():
            if not v:
                continue
            tag = max(v.items(), key=lambda kv: kv[1])[0]
            cur = self.assigned.get(r)
            if cur is None or tag == cur:
                self.pending_change.pop(r, None)
                continue
            prev, n = self.pending_change.get(r, (tag, 0))
            n = n + 1 if prev == tag else 1
            self.pending_change[r] = (tag, n)
            if n >= 2:
                self.assigned[r] = tag
                self.reorders += 1
                self.pending_change.pop(r, None)
        self.epoch_votes = {}

    def tag(self, row: int) -> str:
        return self.assigned.get(row) or f"ROW{row + 1}"

    def resolved(self, row: int) -> bool:
        return row in self.assigned

    def confidence(self, row: int) -> float:
        v = self.votes.get(row)
        if not v:
            return 0.0
        total = sum(v.values())
        tag = self.assigned.get(row)
        return (v.get(tag, 0.0) / total) if (total and tag) else 0.0


@dataclass
class RunStats:
    samples: int = 0
    base_samples: int = 0
    burst_samples: int = 0
    change_points: int = 0
    events: int = 0
    unknown_events: int = 0
    suppressed_feed: int = 0
    tag_rows_resolved: int = 0
    panel_reorders: int = 0
    attributed_kills: int = 0
    seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _slot_key(row: int, slot: int) -> str:
    return f"{row}:{slot}"


class Analyzer:
    def __init__(self, prof: Profile, our_team: Optional[str] = None,
                 workers: int = 4, verbose: bool = False):
        self.prof = prof
        self.our_team = our_team or prof.our_team
        self.workers = max(1, workers)
        self.verbose = verbose
        self.events: List[Event] = []
        self.stats = RunStats()

    # ---------------- pass A: find change points cheaply ----------------
    def scan_changes(self, path: str, info: sampler.VideoInfo, prof: Profile) -> List[float]:
        idx = sampler.base_schedule(info, float(prof.sampling.get("base_fps", 2.0)))
        self.stats.base_samples = len(idx)
        last: Dict[str, Any] = {}
        changes: List[float] = []
        for fi, t, frame in sampler.iter_frames(path, idx):
            vals = self._read_counters(frame, prof)
            for name, (val, conf) in vals.items():
                if val is None or conf < prof.confidence_threshold:
                    continue
                if name in last and last[name] != val:
                    changes.append(t)
                    break
            for name, (val, conf) in vals.items():
                if val is not None and conf >= prof.confidence_threshold:
                    last[name] = val
        self.stats.change_points = len(changes)
        return changes

    def _read_counters(self, frame: np.ndarray, prof: Profile) -> Dict[str, Tuple[Optional[int], float]]:
        names = [n for n in (TEAMS_ALIVE, PLAYERS_ALIVE, ZONE_PHASE) if n in prof.rois]
        if self.workers > 1 and len(names) > 1:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(names))) as ex:
                res = list(ex.map(lambda n: (n, hud.read_counter(frame, prof, n)), names))
            return dict(res)
        return {n: hud.read_counter(frame, prof, n) for n in names}

    # ---------------- pass B: full extraction ----------------
    def run(self, path: str) -> Tuple[List[Event], RunStats]:
        t0 = time.time()
        info = sampler.probe(path)
        prof = self.prof.for_frame_size(info.width, info.height)
        if self.verbose:
            print(f"[analyze] {info.width}x{info.height} {info.fps:.2f}fps "
                  f"{info.frame_count} frames ({info.duration:.1f}s), profile '{prof.name}'")

        if not prof.calibrated:
            print(f"[analyze] WARNING: profile '{prof.name}' is a template "
                  f"(_calibrated=false). Its ROI boxes are placeholders; run "
                  f"'calibrate' on this footage before trusting any output.")
        changes = self.scan_changes(path, info, prof)
        base = sampler.base_schedule(info, float(prof.sampling.get("base_fps", 2.0)))
        burst = sampler.burst_schedule(info, changes,
                                       float(prof.sampling.get("burst_fps", 5.0)),
                                       float(prof.sampling.get("burst_window_s", 3.0)))
        schedule = sampler.merge_schedules(base, burst)
        self.stats.burst_samples = len(set(burst) - set(base))
        if self.verbose:
            print(f"[analyze] {len(changes)} change points -> {len(schedule)} samples "
                  f"({self.stats.burst_samples} burst)")

        panel = prof.rois.get("team_panel")
        rows = int(panel.meta.get("rows", 25)) if panel else 0
        slots = int(panel.meta.get("slots", 4)) if panel else 0
        tags = PanelTagResolver(rows, prof.team_tags)
        if panel:
            self._resolve_tags(path, info, prof, tags,
                               int(prof.sampling.get("tag_prepass_frames", 6)))
            self.stats.tag_rows_resolved = len(tags.assigned)
            if self.verbose:
                print(f"[analyze] panel tags resolved for {len(tags.assigned)}/{rows} rows")

        counters = {
            TEAMS_ALIVE: ConfirmedField(TEAMS_ALIVE, prof.confirm_n(TEAMS_ALIVE),
                                        prof.confidence_threshold, monotonic="down",
                                        reset_at=int(prof.rois[TEAMS_ALIVE].meta.get("reset_at", 20))
                                        if TEAMS_ALIVE in prof.rois else None,
                                        event_type=TEAMS_ALIVE),
            PLAYERS_ALIVE: ConfirmedField(PLAYERS_ALIVE, prof.confirm_n(PLAYERS_ALIVE),
                                          prof.confidence_threshold, monotonic="down",
                                          reset_at=int(prof.rois[PLAYERS_ALIVE].meta.get("reset_at", 70))
                                          if PLAYERS_ALIVE in prof.rois else None,
                                          event_type=PLAYERS_ALIVE),
            ZONE_PHASE: ConfirmedField(ZONE_PHASE, prof.confirm_n(ZONE_PHASE),
                                       prof.confidence_threshold, event_type=ZONE_PHASE),
        }
        zone_state = ConfirmedField("zone_state", prof.confirm_n("zone_state"),
                                    prof.confidence_threshold, event_type=ZONE_STATE)
        slot_fields: Dict[str, ConfirmedField] = {}
        for r in range(rows):
            for s in range(slots):
                slot_fields[_slot_key(r, s)] = ConfirmedField(
                    _slot_key(r, s), prof.confirm_n("slot"), prof.confidence_threshold,
                    event_type="slot", emit_unknown=False)

        feed = FeedTracker(prof, confirm_sightings=int(prof.confirm_samples.get("feed_sightings", 2)))
        slot_transitions: Dict[Tuple[str, int], List[float]] = {}
        fuser = Fuser(fuse_window_s=prof.fuse_window_s,
                      panel_is_authority=bool(panel),
                      panel_blind=lambda team, slot: self._panel_blind(
                          tags, slot_fields, rows, slots, team, slot))

        team_counts: Dict[int, int] = {}
        match_index = 0
        match_open = False
        map_done = False
        our_dead_at: Optional[float] = None
        placement_emitted = False
        winner_reads: List[Tuple[float, Optional[int], str, float]] = []
        last_tag_read = -1e9
        tag_refresh = float(prof.sampling.get("tag_refresh_s", 20.0))
        match_start_t = 0.0

        for fi, t, frame in sampler.iter_frames(path, schedule):
            self.stats.samples += 1
            cvals = self._read_counters(frame, prof)
            read_tags = (t - last_tag_read) >= tag_refresh
            if read_tags:
                last_tag_read = t
                tags.end_epoch()
            pan = hud.read_team_panel(frame, prof, read_tags=read_tags) if panel else None
            if pan:
                for r, (tag, conf) in pan.tags.items():
                    tags.vote(r, tag, conf)

            # --- lobby counters ---
            prev_teams = counters[TEAMS_ALIVE].current
            for name, (val, conf) in cvals.items():
                ev = counters[name].observe(t, fi, val, conf)
                if ev is not None:
                    ev.match_index = match_index
                    self._emit(ev)
            ta = counters[TEAMS_ALIVE].current

            # --- match boundaries ---
            if isinstance(ta, int):
                if not match_open:
                    match_open = True
                    match_start_t = t
                    self._emit(Event(t=t, type=MATCH_START, confidence=counters[TEAMS_ALIVE].current_conf,
                                     frame=fi, source="counters", match_index=match_index,
                                     value=ta))
                elif isinstance(prev_teams, int) and ta > prev_teams:
                    self._finalize_winner(winner_reads, tags, slot_fields, rows, slots,
                                          match_index, t, prof)
                    match_index += 1
                    match_start_t = t
                    winner_reads = []
                    our_dead_at, placement_emitted, map_done = None, False, False
                    team_counts.clear()
                    self._emit(Event(t=t, type=MATCH_START, confidence=counters[TEAMS_ALIVE].current_conf,
                                     frame=fi, source="counter_reset", match_index=match_index,
                                     value=ta))

            # --- map name ---
            if not map_done and (t - match_start_t) <= float(prof.sampling.get("map_read_window_s", 90.0)):
                name, conf = hud.read_choice_text(frame, prof, "map_name", prof.map_names)
                if name and conf >= max(prof.confidence_threshold, 0.7):
                    self._emit(Event(t=t, type=MAP_NAME, confidence=conf, frame=fi,
                                     source="ocr", value=name, match_index=match_index))
                    map_done = True

            # --- zone state ---
            zs, zconf = hud.read_zone_state(frame, prof)
            ev = zone_state.observe(t, fi, zs, zconf)
            if ev is not None:
                ev.match_index = match_index
                ev.zone_phase = counters[ZONE_PHASE].current if isinstance(
                    counters[ZONE_PHASE].current, int) else None
                self._emit(ev)

            # --- team panel: per-slot transitions -> knock/kill ---
            if pan:
                for sr in pan.slots:
                    key = _slot_key(sr.team_row, sr.slot)
                    f = slot_fields[key]
                    before = f.current
                    tev = f.observe(t, fi, sr.state, sr.confidence)
                    if tev is None:
                        continue
                    after = f.current
                    kind = self._transition_kind(before, after)
                    if kind is None:
                        continue
                    slot_transitions.setdefault(
                        (tags.tag(sr.team_row), sr.slot + 1), []).append(tev.t)
                    fuser.add_panel(PanelEvent(
                        t=tev.t, frame=tev.frame or fi, kind=kind,
                        victim_team=tags.tag(sr.team_row), victim_slot=sr.slot + 1,
                        confidence=tev.confidence,
                        zone_phase=counters[ZONE_PHASE].current if isinstance(
                            counters[ZONE_PHASE].current, int) else None,
                        match_index=match_index))
                # per-team alive count
                for r in range(rows):
                    states = [slot_fields[_slot_key(r, s)].current for s in range(slots)]
                    if any(st in (None, UNKNOWN) for st in states):
                        continue
                    alive = sum(1 for st in states if st != hud.DEAD)
                    if team_counts.get(r) != alive:
                        team_counts[r] = alive
                        confs = [slot_fields[_slot_key(r, s)].current_conf for s in range(slots)]
                        self._emit(Event(t=t, type=TEAM_ALIVE_COUNT, confidence=min(confs),
                                         frame=fi, source="panel", team=tags.tag(r),
                                         value=alive, match_index=match_index,
                                         zone_phase=counters[ZONE_PHASE].current if isinstance(
                                             counters[ZONE_PHASE].current, int) else None))
                    if (self.our_team and tags.tag(r) == self.our_team and alive == 0
                            and our_dead_at is None):
                        our_dead_at = t

            # --- kill feed ---
            for fe in feed.observe(t, hud.read_feed_lines(frame, prof)):
                fuser.add_feed(fe)
            for ev in fuser.flush(now=t):
                ev.match_index = match_index
                self._emit(ev)

            # --- our placement / winner, read near the end of a match ---
            near_end = isinstance(ta, int) and ta <= int(prof.sampling.get("endgame_teams", 10))
            if not placement_emitted and (our_dead_at is not None or near_end):
                val, conf = hud.read_placement(frame, prof)
                if val is not None and conf >= prof.confidence_threshold:
                    self._emit(Event(t=t, type=PLACEMENT, confidence=conf, frame=fi,
                                     source="ocr", value=val, team=self.our_team,
                                     match_index=match_index))
                    placement_emitted = True
                elif our_dead_at is not None and isinstance(ta, int):
                    # fall back to the lobby counter at the moment we were wiped
                    self._emit(Event(t=our_dead_at, type=PLACEMENT,
                                     confidence=min(0.85, counters[TEAMS_ALIVE].current_conf),
                                     frame=fi, source="teams_alive_at_wipe", value=ta,
                                     team=self.our_team, match_index=match_index))
                    placement_emitted = True
            # The winner is decided once, at the end of the match: the banner
            # is the evidence, and the panel's last-standing row is the
            # fallback. Committing the moment teams_alive hits 1 reads a panel
            # that has not finished confirming the last deaths.
            if isinstance(ta, int) and ta <= 2:
                wt, wconf = hud.read_winner_tag(frame, prof)
                if wt is not None and wconf >= prof.confidence_threshold:
                    winner_reads.append((t, fi, wt, wconf))

        for ev in fuser.flush(now=None):
            ev.match_index = match_index
            self._emit(ev)
        if match_open:
            self._finalize_winner(winner_reads, tags, slot_fields, rows, slots,
                                  match_index, info.duration, prof)

        self.stats.suppressed_feed = len(fuser.suppressed_feed)
        self.stats.panel_reorders = tags.reorders
        self.stats.attributed_kills = fuser.attributed
        self.stats.seconds = time.time() - t0
        self.events.sort(key=lambda e: (e.t, e.type, str(e.victim_team), e.victim_slot or 0))
        self.stats.events = len(self.events)
        self.stats.unknown_events = sum(1 for e in self.events if e.state == "unknown")
        return self.events, self.stats

    # ---------------- helpers ----------------
    def _finalize_winner(self, winner_reads: List[Tuple[float, Optional[int], str, float]],
                         tags: PanelTagResolver, slot_fields: Dict[str, ConfirmedField],
                         rows: int, slots: int, match_index: int, end_t: float,
                         prof: Profile) -> None:
        """Decide the winner from the banner, else from the last row standing."""
        if winner_reads:
            weight: Dict[str, float] = {}
            for _, _, tag, conf in winner_reads:
                weight[tag] = weight.get(tag, 0.0) + conf
            tag = max(weight.items(), key=lambda kv: kv[1])[0]
            best = max((w for w in winner_reads if w[2] == tag), key=lambda w: w[3])
            t, fi, _, conf = best
            self._emit(Event(t=t, type=WINNER, confidence=conf, frame=fi,
                             source="banner", value=tag, match_index=match_index))
            self._emit(Event(t=t, type=MATCH_END, confidence=conf, frame=fi,
                             source="winner", match_index=match_index))
            return
        live = [r for r in range(rows)
                if any(slot_fields[_slot_key(r, s)].current not in (hud.DEAD, None, UNKNOWN)
                       for s in range(slots))]
        if len(live) == 1 and tags.resolved(live[0]):
            conf = min(0.8, max(0.6, tags.confidence(live[0])))
            self._emit(Event(t=end_t, type=WINNER, confidence=conf, source="panel_last_standing",
                             value=tags.tag(live[0]), match_index=match_index))
            self._emit(Event(t=end_t, type=MATCH_END, confidence=conf, source="winner",
                             match_index=match_index))
            return
        self._emit(Event(t=end_t, type=WINNER, confidence=0.0, state="unknown",
                         source="no_banner_no_single_row", match_index=match_index))
        self._emit(Event(t=end_t, type=MATCH_END, confidence=0.5, state="unknown",
                         source="end_of_video", match_index=match_index))

    def _resolve_tags(self, path: str, info: sampler.VideoInfo, prof: Profile,
                      tags: PanelTagResolver, n_frames: int) -> None:
        """Pre-pass: read the panel's team tags on a few spread-out frames."""
        if info.frame_count <= 0:
            return
        step = max(1, info.frame_count // max(1, n_frames + 1))
        idx = [min(info.frame_count - 1, step * (i + 1)) for i in range(n_frames)]
        for fi, t, frame in sampler.iter_frames(path, idx):
            pan = hud.read_team_panel(frame, prof, read_tags=True)
            for r, (tag, conf) in pan.tags.items():
                tags.vote(r, tag, conf)
        tags.resolve()

    def _emit(self, ev: Event) -> None:
        self.events.append(ev)

    @staticmethod
    def _transition_kind(before: Any, after: Any) -> Optional[str]:
        if before in (None, UNKNOWN):
            return None   # first sighting of a slot is not an event
        if before == hud.ALIVE and after == hud.KNOCKED:
            return KNOCK
        if before in (hud.ALIVE, hud.KNOCKED) and after == hud.DEAD:
            return KILL
        return None

    @staticmethod
    def _panel_blind(tags: PanelTagResolver, slot_fields: Dict[str, ConfirmedField],
                     rows: int, slots: int, team: Optional[str],
                     slot: Optional[int]) -> bool:
        """True when the panel holds no confirmed state for this victim.

        Only then may a feed line stand on its own: the panel has nothing to
        say about that slot, so the feed is the only witness there is.
        """
        if team is None:
            return False
        for r in range(rows):
            if tags.tag(r) != team:
                continue
            idxs = range(slots) if slot is None else [int(slot) - 1]
            for s in idxs:
                f = slot_fields.get(_slot_key(r, int(s)))
                if f is not None and f.current not in (None, UNKNOWN):
                    return False
        return True


def analyze_video(video: str, profile: Profile, out_path: Optional[str] = None,
                  our_team: Optional[str] = None, workers: int = 4,
                  verbose: bool = False) -> Tuple[List[Event], RunStats]:
    an = Analyzer(profile, our_team=our_team, workers=workers, verbose=verbose)
    events, stats = an.run(video)
    if out_path:
        from .events import JsonlWriter
        with JsonlWriter(out_path) as w:
            w.write_all(events)
    return events, stats
