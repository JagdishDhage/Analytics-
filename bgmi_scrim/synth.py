"""Synthetic spectator-HUD footage with exact ground-truth labels.

This is test scaffolding, not an analysis component. It renders a HUD with the
same geometry as ``profiles/synthetic_*.json`` and writes the labels that the
ground-truth harness scores against, so the extractor's precision and recall
can be measured without hand-labelling. Numbers measured on it are numbers on
synthetic footage; they are not a substitute for measuring on real scrim VODs.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ALIVE, KNOCKED, DEAD = "alive", "knocked", "dead"

COL_ALIVE = (60, 200, 60)
COL_KNOCKED = (0, 140, 255)
COL_DEAD = (70, 70, 70)
COL_ZONE_SHRINK = (235, 150, 40)
COL_ZONE_WAIT = (225, 225, 225)
COL_TEXT = (235, 235, 235)
COL_BG = (18, 18, 20)
COL_PANEL = (32, 32, 36)

TAGS_POOL = ["TX", "GB", "SOUL", "GODL", "ENT", "OR", "BLIND", "HYD", "REV", "SKY",
             "VLT", "NGE", "MARC", "AUT", "VST", "TSM", "XSP", "IQOO", "ORCA", "ZEUS",
             "NOVA", "FS", "GXR", "MDY", "K9"]


@dataclass
class Layout:
    width: int = 1920
    height: int = 1080

    @property
    def s(self) -> float:
        return self.width / 1920.0

    def sc(self, v: float) -> int:
        return int(round(v * self.s))

    # HUD boxes in 1920x1080 reference pixels
    teams_box = (1700, 28, 60, 44)
    players_box = (1820, 28, 70, 44)
    panel_origin = (1640, 120)
    panel_rows = 25
    row_pitch = 34
    tag_box = (0, 2, 72, 26)
    slot_x0 = 82
    slot_pitch = 26
    slot_wh = (18, 18)
    feed_origin = (620, 92)
    feed_rows = 5
    feed_row_pitch = 34
    feed_row_wh = (540, 28)
    zone_state_box = (70, 980, 36, 36)
    zone_phase_box = (120, 980, 44, 40)
    map_box = (60, 28, 210, 42)
    placement_box = (900, 470, 130, 84)
    winner_box = (700, 200, 520, 72)


@dataclass
class Player:
    team: int
    slot: int
    state: str = ALIVE


@dataclass
class Sim:
    n_teams: int = 25
    seed: int = 7
    duration: float = 150.0
    our_team: str = "TX"
    map_name: str = "ERANGEL"
    tags: List[str] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)

    def build(self) -> "Sim":
        rng = random.Random(self.seed)
        self.tags = TAGS_POOL[:self.n_teams]
        players = {(ti, s): Player(ti, s) for ti in range(self.n_teams) for s in range(4)}
        events: List[Dict[str, Any]] = []
        t = 8.0
        # zone phases across the match
        phases = []
        pt = 10.0
        for ph in range(1, 9):
            phases.append({"t": pt, "phase": ph, "state": "waiting"})
            phases.append({"t": pt + 6.0, "phase": ph, "state": "shrinking"})
            pt += 16.0
        pending: List[Tuple[float, Tuple[int, int], Tuple[int, int]]] = []
        # pace knocks so that exactly one team is left shortly before the end
        need = int(self.n_teams * 4 * 1.3)
        gap = max(0.35, (self.duration - 20.0) / max(1, need))
        while t < self.duration - 10:
            for item in list(pending):
                due, victim, killer = item
                if due <= t:
                    pending.remove(item)
                    if players[victim].state == KNOCKED:
                        if rng.random() < 0.15:
                            players[victim].state = ALIVE          # revived
                            events.append({"t": round(due, 2), "type": "revive",
                                           "victim_team": self.tags[victim[0]],
                                           "victim_slot": victim[1] + 1})
                        else:
                            players[victim].state = DEAD
                            events.append({"t": round(due, 2), "type": "kill",
                                           "victim_team": self.tags[victim[0]],
                                           "victim_slot": victim[1] + 1,
                                           "killer_team": self.tags[killer[0]],
                                           "killer_slot": killer[1] + 1})
            alive_teams = [ti for ti in range(self.n_teams)
                           if any(players[(ti, s)].state != DEAD for s in range(4))]
            if len(alive_teams) <= 1 and not pending:
                break
            cands = [(ti, s) for ti in alive_teams for s in range(4)
                     if players[(ti, s)].state == ALIVE]
            killers_all = [c for c in cands]
            if len(alive_teams) > 1 and cands:
                # focus fire: a team already down a player is likelier to be pushed
                def weight(c):
                    dead = sum(1 for s2 in range(4) if players[(c[0], s2)].state == DEAD)
                    return 1.0 + 1.6 * dead
                victim = rng.choices(cands, weights=[weight(c) for c in cands], k=1)[0]
                killers = [c for c in killers_all if c[0] != victim[0]]
                if killers:
                    killer = rng.choice(killers)
                    players[victim].state = KNOCKED
                    events.append({"t": round(t, 2), "type": "knock",
                                   "victim_team": self.tags[victim[0]],
                                   "victim_slot": victim[1] + 1,
                                   "killer_team": self.tags[killer[0]],
                                   "killer_slot": killer[1] + 1})
                    pending.append((t + rng.uniform(1.5, 5.0), victim, killer))
            t += rng.uniform(0.6 * gap, 1.4 * gap)
        # finish anything still knocked, and wipe any leftover teams but the last
        for due, victim, killer in list(pending):
            if players[victim].state == KNOCKED:
                players[victim].state = DEAD
                events.append({"t": round(min(due, self.duration - 9), 2), "type": "kill",
                               "victim_team": self.tags[victim[0]], "victim_slot": victim[1] + 1,
                               "killer_team": self.tags[killer[0]], "killer_slot": killer[1] + 1})
        alive_teams = [ti for ti in range(self.n_teams)
                       if any(players[(ti, s)].state != DEAD for s in range(4))]
        if len(alive_teams) > 1:
            survivor = rng.choice(alive_teams)
            tt = self.duration - 9.0
            for ti in alive_teams:
                if ti == survivor:
                    continue
                for sl in range(4):
                    if players[(ti, sl)].state != DEAD:
                        players[(ti, sl)].state = DEAD
                        events.append({"t": round(tt, 2), "type": "kill",
                                       "victim_team": self.tags[ti], "victim_slot": sl + 1,
                                       "killer_team": self.tags[survivor],
                                       "killer_slot": rng.randint(1, 4)})
                        tt += 0.6
        events.sort(key=lambda e: e["t"])
        self.timeline = events
        self.phases = phases
        return self

    # ---- replay state at a time ----
    def state_at(self, t: float) -> Dict[str, Any]:
        states = {(ti, s): ALIVE for ti in range(self.n_teams) for s in range(4)}
        feed: List[Dict[str, Any]] = []
        for e in self.timeline:
            if e["t"] > t:
                break
            ti = self.tags.index(e["victim_team"])
            s = e["victim_slot"] - 1
            if e["type"] == "knock":
                states[(ti, s)] = KNOCKED
            elif e["type"] == "kill":
                states[(ti, s)] = DEAD
            elif e["type"] == "revive":
                states[(ti, s)] = ALIVE
        for e in self.timeline:
            if e["type"] == "revive":
                continue
            if e["t"] <= t <= e["t"] + 6.0:
                feed.append(e)
        phase, zstate = 1, "waiting"
        for p in self.phases:
            if p["t"] <= t:
                phase, zstate = p["phase"], p["state"]
        players_alive = sum(1 for v in states.values() if v != DEAD)
        teams_alive = sum(1 for ti in range(self.n_teams)
                          if any(states[(ti, s)] != DEAD for s in range(4)))
        team_alive = {self.tags[ti]: sum(1 for s in range(4) if states[(ti, s)] != DEAD)
                      for ti in range(self.n_teams)}
        return {"states": states, "feed": feed[-5:], "zone_phase": phase,
                "zone_state": zstate, "players_alive": players_alive,
                "teams_alive": teams_alive, "team_alive": team_alive}

    def our_placement(self) -> Tuple[Optional[float], Optional[int]]:
        """(time we were wiped, placement) -- placement = teams alive at that moment."""
        for e in self.timeline:
            if e["type"] != "kill" or e["victim_team"] != self.our_team:
                continue
            st = self.state_at(e["t"] + 0.01)
            if st["team_alive"].get(self.our_team, 0) == 0:
                return e["t"], st["teams_alive"]
        return None, None

    def winner(self) -> Optional[str]:
        st = self.state_at(self.duration)
        live = [tag for tag, n in st["team_alive"].items() if n > 0]
        return live[0] if len(live) == 1 else None


def _put(img: np.ndarray, text: str, org: Tuple[int, int], scale: float,
         colour=COL_TEXT, thick: int = 2) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def render_frame(sim: Sim, t: float, lay: Layout, noise: float = 0.0,
                 rng: Optional[random.Random] = None) -> np.ndarray:
    st = sim.state_at(t)
    s = lay.s
    img = np.full((lay.height, lay.width, 3), COL_BG, np.uint8)
    # faint background texture so the HUD is not on a flat plate
    cv2.rectangle(img, (0, 0), (lay.width, lay.height), (26, 30, 26), -1)
    for i in range(0, lay.width, 160):
        cv2.line(img, (i, 0), (i + 90, lay.height), (22, 26, 22), 2)

    def box(b):
        x, y, w, h = b
        return (int(x * s), int(y * s), int(w * s), int(h * s))

    # counters
    for label, b, val in (("TEAM", lay.teams_box, st["teams_alive"]),
                          ("ALIVE", lay.players_box, st["players_alive"])):
        x, y, w, h = box(b)
        cv2.rectangle(img, (x - int(6 * s), y - int(6 * s)),
                      (x + w + int(6 * s), y + h + int(6 * s)), COL_PANEL, -1)
        _put(img, str(val), (x + int(4 * s), y + int(h * 0.82)), 1.0 * s, COL_TEXT, max(1, int(2 * s)))
        _put(img, label, (x, y - int(10 * s)), 0.4 * s, (150, 150, 150), 1)

    # map name
    x, y, w, h = box(lay.map_box)
    _put(img, sim.map_name, (x, y + int(h * 0.78)), 0.8 * s, COL_TEXT, max(1, int(2 * s)))

    # team panel
    px, py = int(lay.panel_origin[0] * s), int(lay.panel_origin[1] * s)
    cv2.rectangle(img, (px - int(8 * s), py - int(8 * s)),
                  (px + int(190 * s), py + int(lay.panel_rows * lay.row_pitch * s)),
                  COL_PANEL, -1)
    for r in range(lay.panel_rows):
        ry = py + int(r * lay.row_pitch * s)
        tb = lay.tag_box
        _put(img, sim.tags[r], (px + int(tb[0] * s), ry + int((tb[1] + 21) * s)),
             0.55 * s, COL_TEXT, max(1, int(2 * s)))
        for sl in range(4):
            sx = px + int((lay.slot_x0 + sl * lay.slot_pitch) * s)
            sy = ry + int(4 * s)
            w_, h_ = int(lay.slot_wh[0] * s), int(lay.slot_wh[1] * s)
            state = st["states"][(r, sl)]
            col = {ALIVE: COL_ALIVE, KNOCKED: COL_KNOCKED, DEAD: COL_DEAD}[state]
            cv2.rectangle(img, (sx, sy), (sx + w_, sy + h_), col, -1)

    # kill feed
    fx, fy = int(lay.feed_origin[0] * s), int(lay.feed_origin[1] * s)
    for i, e in enumerate(st["feed"][::-1]):
        ry = fy + int(i * lay.feed_row_pitch * s)
        w_, h_ = int(lay.feed_row_wh[0] * s), int(lay.feed_row_wh[1] * s)
        cv2.rectangle(img, (fx, ry), (fx + w_, ry + h_), COL_PANEL, -1)
        verb = "KNOCKED" if e["type"] == "knock" else "FINISHED"
        txt = f"{e.get('killer_team','')}{e.get('killer_slot','')} {verb} {e['victim_team']}{e['victim_slot']}"
        _put(img, txt, (fx + int(6 * s), ry + int(h_ * 0.78)), 0.6 * s, COL_TEXT, max(1, int(2 * s)))

    # zone indicator + phase
    x, y, w, h = box(lay.zone_state_box)
    col = COL_ZONE_SHRINK if st["zone_state"] == "shrinking" else COL_ZONE_WAIT
    cv2.rectangle(img, (x, y), (x + w, y + h), col, -1)
    x, y, w, h = box(lay.zone_phase_box)
    _put(img, str(st["zone_phase"]), (x + int(6 * s), y + int(h * 0.82)), 0.9 * s,
         COL_TEXT, max(1, int(2 * s)))
    _put(img, "PHASE", (x - int(4 * s), y - int(8 * s)), 0.4 * s, (150, 150, 150), 1)

    # placement card for our team, and the winner banner
    wipe_t, placement = sim.our_placement()
    if wipe_t is not None and placement is not None and wipe_t <= t <= wipe_t + 8.0:
        x, y, w, h = box(lay.placement_box)
        cv2.rectangle(img, (x - int(20 * s), y - int(20 * s)),
                      (x + w + int(20 * s), y + h + int(20 * s)), (40, 40, 48), -1)
        _put(img, f"{placement}", (x + int(10 * s), y + int(h * 0.85)), 2.0 * s, COL_TEXT,
             max(2, int(4 * s)))
        _put(img, "RANK", (x, y - int(26 * s)), 0.6 * s, (170, 170, 170), max(1, int(2 * s)))
    win = sim.winner()
    if win and t >= sim.duration - 8.0:
        x, y, w, h = box(lay.winner_box)
        cv2.rectangle(img, (x, y), (x + w, y + h), (45, 45, 30), -1)
        _put(img, win, (x + int(16 * s), y + int(h * 0.76)), 1.4 * s, COL_TEXT, max(2, int(3 * s)))

    if noise > 0:
        rng = rng or random.Random(0)
        n = np.random.default_rng(rng.randint(0, 1 << 30)).normal(
            0, noise * 255, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0.6)
    return img


def labels_for(sim: Sim, label_fps: float = 2.0) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    out.append({"kind": "scalar", "t": 1.0, "type": "map_name", "value": sim.map_name})
    for e in sim.timeline:
        if e["type"] == "revive":
            continue
        out.append({"kind": "event", **e})
    step = 1.0 / label_fps
    t = 0.0
    while t < sim.duration:
        st = sim.state_at(t)
        out.append({"kind": "frame", "t": round(t, 3), "teams_alive": st["teams_alive"],
                    "players_alive": st["players_alive"], "zone_phase": st["zone_phase"],
                    "zone_state": st["zone_state"], "team_alive": st["team_alive"]})
        t += step
    wipe_t, placement = sim.our_placement()
    if placement is not None:
        out.append({"kind": "scalar", "t": wipe_t, "type": "placement", "value": placement,
                    "team": sim.our_team})
    win = sim.winner()
    if win:
        out.append({"kind": "scalar", "t": sim.duration - 4.0, "type": "winner", "value": win})
    return out


def write_clip(out_video: str, out_labels: str, width: int = 1920, height: int = 1080,
               fps: float = 30.0, duration: float = 150.0, seed: int = 7,
               noise: float = 0.0, our_team: str = "TX", map_name: str = "ERANGEL",
               n_teams: int = 25) -> Dict[str, Any]:
    lay = Layout(width, height)
    sim = Sim(n_teams=n_teams, seed=seed, duration=duration, our_team=our_team,
              map_name=map_name).build()
    Path(out_video).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not vw.isOpened():
        raise IOError("cannot open VideoWriter")
    rng = random.Random(seed)
    n_frames = int(duration * fps)
    for i in range(n_frames):
        vw.write(render_frame(sim, i / fps, lay, noise=noise, rng=rng))
    vw.release()
    labels = labels_for(sim)
    Path(out_labels).write_text("\n".join(json.dumps(x, sort_keys=True) for x in labels) + "\n")
    return {"video": out_video, "labels": out_labels, "events": len(sim.timeline),
            "tags": sim.tags, "our_team": our_team,
            "placement": sim.our_placement()[1], "winner": sim.winner()}
