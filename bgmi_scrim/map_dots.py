"""Finding player dots on the map and reading the name beside them."""
from __future__ import annotations

import difflib
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .map_profile import MapProfile
from .ocr import ALNUM, read_text
from .preprocess import prepare_roi, roi_is_blank


@dataclass
class ColourBook:
    """What colour each team's dots are, learned as labels are confirmed.

    Observer maps colour-code teams, so the dot's own colour is a second
    witness to a label: it is read from different pixels than the text and
    fails in different ways. A label that disagrees with a team's established
    colour is dropped rather than averaged in, which is what stops a name read
    off a neighbouring dot in a crowded corner from moving a whole track.
    """

    tol_deg: float = 12.0
    min_observations: int = 3
    hues: Dict[str, List[float]] = field(default_factory=dict)

    @staticmethod
    def hue_of(colour: Tuple[int, int, int]) -> Optional[float]:
        b, g, r = [int(c) for c in colour]
        px = np.uint8([[[b, g, r]]])
        hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]
        if int(hsv[1]) < 60 or int(hsv[2]) < 60:
            return None
        return float(hsv[0]) * 2.0            # OpenCV hue is 0..179

    def observe(self, team: str, colour: Tuple[int, int, int]) -> None:
        h = self.hue_of(colour)
        if h is None:
            return
        self.hues.setdefault(team, []).append(h)
        if len(self.hues[team]) > 60:
            self.hues[team] = self.hues[team][-60:]

    def known(self, team: str) -> bool:
        return len(self.hues.get(team, [])) >= self.min_observations

    def mean_hue(self, team: str) -> Optional[float]:
        hs = self.hues.get(team)
        if not hs:
            return None
        ang = np.radians(np.array(hs))
        return float((math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean())) + 360) % 360)

    def team_from_colour(self, colour: Tuple[int, int, int]) -> Tuple[Optional[str], float]:
        """Nearest known team hue, when it is clearly nearer than the next one."""
        h = self.hue_of(colour)
        if h is None:
            return None, 0.0
        ds = []
        for team in self.hues:
            if not self.known(team):
                continue
            m = self.mean_hue(team)
            if m is None:
                continue
            d = abs(h - m) % 360.0
            ds.append((min(d, 360.0 - d), team))
        if not ds:
            return None, 0.0
        ds.sort()
        best_d, team = ds[0]
        if best_d > self.tol_deg:
            return None, 0.0
        margin = (ds[1][0] - best_d) if len(ds) > 1 else 180.0
        # With 25 teams sharing the hue circle, "nearest hue" is not an
        # identification unless the runner-up is clearly further away.
        if margin < self.tol_deg:
            return None, 0.0
        conf = min(0.85, 0.4 + 0.3 * (1.0 - best_d / max(self.tol_deg, 1e-6))
                   + 0.3 * min(1.0, margin / (3 * self.tol_deg)))
        return team, conf

    def agrees(self, team: str, colour: Tuple[int, int, int]) -> Optional[bool]:
        """True/False when the team's colour is known, None when it is not."""
        if not self.known(team):
            return None
        h = self.hue_of(colour)
        m = self.mean_hue(team)
        if h is None or m is None:
            return None
        d = abs(h - m) % 360.0
        return min(d, 360.0 - d) <= self.tol_deg


@dataclass
class Dot:
    px: float
    py: float
    area: float
    radius: float
    colour: Tuple[int, int, int]        # mean BGR
    cluster_size: int = 1               # >1 when several dots have merged
    confidence: float = 0.0
    team: Optional[str] = None
    player: Optional[str] = None
    team_conf: float = 0.0              # how sure we are of the team
    label_conf: float = 0.0             # how sure we are of the player
    label_raw: str = ""
    team_source: str = ""               # label | colour


def _mask(frame: np.ndarray, prof: MapProfile) -> np.ndarray:
    x, y, w, h = [int(v) for v in prof.map_area]
    sub = frame[max(0, y):y + h, max(0, x):x + w]
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    sat_min = int(prof.dots.get("sat_min", 90))
    val_min = int(prof.dots.get("val_min", 90))
    m = cv2.inRange(hsv, np.array([0, sat_min, val_min], np.uint8),
                    np.array([179, 255, 255], np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def detect_dots(frame: np.ndarray, prof: MapProfile) -> List[Dot]:
    """Saturated, roughly circular blobs inside the map viewport.

    A blob much bigger than one dot is players standing on each other, which
    is normal in a squad game: it is reported once with ``cluster_size`` set
    rather than split into invented positions.
    """
    x0, y0, _, _ = [int(v) for v in prof.map_area]
    m = _mask(frame, prof)
    n, _, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    min_a = float(prof.dots.get("min_area", 14))
    max_a = float(prof.dots.get("max_area", 400))
    single = float(prof.dots.get("single_area", 60.0))
    ratio = float(prof.dots.get("cluster_ratio", 1.6))
    max_cluster = int(prof.dots.get("max_cluster", 4))
    out: List[Dot] = []
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if area < min_a or area > max_a * max_cluster:
            continue
        # circularity: a zone ring is long and thin, a dot is round
        peri = 2.0 * (bw + bh)
        circ = 4.0 * math.pi * float(area) / (peri * peri) if peri else 0.0
        if circ < float(prof.dots.get("min_circularity", 0.45)):
            continue
        if bw > 4 * bh or bh > 4 * bw:
            continue
        cx, cy = cents[i]
        patch = frame[y0 + by:y0 + by + bh, x0 + bx:x0 + bx + bw]
        colour = tuple(int(v) for v in patch.reshape(-1, 3).mean(axis=0)) if patch.size else (0, 0, 0)
        k = 1
        if area > ratio * single:
            k = min(max_cluster, max(1, int(round(float(area) / single))))
        conf = 0.9 if k == 1 else max(0.4, 0.9 - 0.15 * (k - 1))
        out.append(Dot(px=float(x0 + cx), py=float(y0 + cy), area=float(area),
                       radius=math.sqrt(float(area) / math.pi), colour=colour,  # type: ignore[arg-type]
                       cluster_size=k, confidence=conf))
    return _dedup(out, float(prof.dots.get("dedup_px", 3.0)))


def _dedup(dots: List[Dot], px: float) -> List[Dot]:
    out: List[Dot] = []
    for d in sorted(dots, key=lambda d: -d.area):
        if any(math.dist((d.px, d.py), (o.px, o.py)) < px for o in out):
            continue
        out.append(d)
    return out


# ---------------- labels ----------------
def _norm(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def candidate_labels(prof: MapProfile) -> List[Tuple[str, str, str]]:
    """(normalised label, team, player) for every player on the roster."""
    return [(_norm(f"{t}X{p}"), t, p) for t, p in prof.all_labels()]


def match_label(text: str, prof: MapProfile,
                candidates: Optional[Sequence[Tuple[str, str, str]]] = None
                ) -> Tuple[Optional[str], Optional[str], float]:
    """Resolve an OCR'd label to a roster entry.

    Matching the whole string against the roster beats parsing it: "TXxJAGDISH"
    uppercases to "TXXJAGDISH", where no regex can tell the separator from the
    team tag, but the roster can.
    """
    t = _norm(text)
    if len(t) < 3:
        return None, None, 0.0
    cands = list(candidates if candidates is not None else candidate_labels(prof))
    if cands:
        scored = sorted(((difflib.SequenceMatcher(None, t, norm).ratio(), team, player)
                         for norm, team, player in cands), reverse=True)
        best, bteam, bplayer = scored[0]
        if best < float(prof.label.get("min_score", 0.6)):
            return None, None, best
        # A short tag like OR sits inside half the roster once the text is
        # garbled. If a second entry scores almost as well, the label does not
        # say which, so it says nothing.
        margin = float(prof.label.get("min_margin", 0.06))
        runner = next((sc for sc, tm, pl in scored[1:] if tm != bteam), 0.0)
        if best - runner < margin:
            return None, None, best * 0.5
        return bteam, bplayer, best
    m = prof.label_re().search(text.upper())
    if m:
        return m.group("team"), m.group("player"), 0.55
    return None, None, 0.0


def _crowded(d: Dot, others: Sequence[Dot], box: Tuple[float, float, float, float]) -> bool:
    """Is another dot sitting inside this dot's label box?"""
    dx, dy, w, h = box
    x0, y0 = d.px + dx, d.py + dy
    for o in others:
        if o is d:
            continue
        if x0 - 4 <= o.px <= x0 + w + 4 and y0 - 6 <= o.py <= y0 + h + 6:
            return True
    return False


def match_team(text: str, prof: MapProfile) -> Tuple[Optional[str], float]:
    """Find the team tag in an OCR'd label, even a crowded one.

    Four stacked team-mates produce one blob of text like
    "TXXALFATXXBRAVO"; no player can be read out of that, but the team tag is
    in it several times over, and the team is what most of the analysis needs.
    """
    t = _norm(text)
    if len(t) < 3:
        return None, 0.0
    scores: List[Tuple[float, str]] = []
    for team in prof.teams:
        tag = _norm(team)
        if not tag:
            continue
        # "TEAMxPLAYER" normalises to TAG + "X" + NAME, so the tag followed by
        # an X is far more specific than the tag alone -- "OR" matches half the
        # alphabet soup a crowded label produces, "ORX" does not.
        if (tag + "X") in t:
            s = min(1.0, 0.72 + 0.04 * len(tag))
        elif t.startswith(tag):
            s = min(0.9, 0.6 + 0.04 * len(tag))
        else:
            s = difflib.SequenceMatcher(None, t[:len(tag) + 2], tag).ratio() * 0.75
        scores.append((s, team))
    if not scores:
        return None, 0.0
    scores.sort(reverse=True)
    score, best = scores[0]
    runner = scores[1][0] if len(scores) > 1 else 0.0
    if score - runner < float(prof.label.get("min_margin", 0.06)):
        return None, score * 0.5          # two tags fit this text equally well
    # a player name from that team in the same text is a second witness
    if any(_norm(p) in t for p in prof.players_of(best)):
        score = min(1.0, score + 0.12)
    else:
        score *= 0.8
    return (best, score) if score >= 0.66 else (None, score)


def assign_teams_by_colour(dots: Sequence[Dot], colours: "ColourBook",
                           min_conf: float = 0.55) -> None:
    """Give every unnamed dot a team from its colour, where that is known."""
    for d in dots:
        if d.team is not None and d.team_conf >= min_conf:
            continue
        team, conf = colours.team_from_colour(d.colour)
        if team is not None and conf >= min_conf and conf > d.team_conf:
            d.team, d.team_conf, d.team_source = team, conf, "colour"


def read_dot_labels(frame: np.ndarray, prof: MapProfile, dots: Sequence[Dot],
                    workers: int = 4, colours: Optional[ColourBook] = None) -> None:
    """OCR the name beside each dot, in place.

    Two guards keep a name from landing on the wrong dot: a label box with
    another dot inside it is treated as ambiguous and down-weighted, and a
    name whose dot colour disagrees with that team's known colour is dropped.
    """
    lab = prof.label
    dx, dy = float(lab.get("dx", 10)), float(lab.get("dy", -20))
    w, h = int(round(float(lab.get("w", 150)))), int(round(float(lab.get("h", 20))))
    cands = candidate_labels(prof)
    H, W = frame.shape[:2]
    all_dots = list(dots)

    def one(d: Dot) -> None:
        bx = int(round(d.px + dx))
        by = int(round(d.py + dy))
        bx = max(0, min(W - 1, bx))
        by = max(0, min(H - 1, by))
        pre = prepare_roi(frame, (bx, by, w, h), prof.preprocess)
        if roi_is_blank(pre):
            return
        txt, conf = read_text([pre.get("binary_otsu"), pre["binary"]], ALNUM + "_")
        if not txt:
            return
        d.label_raw = txt
        ambiguous = _crowded(d, all_dots, (dx, dy, w, h)) or d.cluster_size > 1
        team, player, score = match_label(txt, prof, cands)
        tteam, tscore = match_team(txt, prof)
        if team is not None and colours is not None and colours.agrees(team, d.colour) is False:
            team, score = None, 0.0           # colour says this name is not ours
        if tteam is not None and colours is not None and colours.agrees(tteam, d.colour) is False:
            tteam, tscore = None, 0.0
        # team first: it survives crowding, and it is what most analysis needs
        if tteam is not None:
            d.team = tteam
            d.team_conf = min(1.0, 0.3 + 0.7 * tscore) * (0.8 if ambiguous else 1.0)
            d.team_source = "label"
        elif team is not None:
            d.team = team
            d.team_conf = min(1.0, 0.3 + 0.7 * score) * (0.8 if ambiguous else 1.0)
            d.team_source = "label"
        strict = float(prof.label.get("player_min_score", 0.88))
        if team is not None and not ambiguous and score >= strict:
            d.player = player
            d.label_conf = min(1.0, 0.25 + 0.75 * score)
            if colours is not None:
                colours.observe(team, d.colour)
        elif d.team is not None and colours is not None and not ambiguous and tscore >= 0.85:
            colours.observe(d.team, d.colour)
        if ambiguous:
            d.label_raw += " (crowded)"

    todo = list(dots)[: int(prof.label.get("max_per_frame", 120))]
    if workers > 1 and len(todo) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, todo))
    else:
        for d in todo:
            one(d)
