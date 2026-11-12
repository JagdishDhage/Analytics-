"""Kill-feed parsing, sighting confirmation and de-duplication.

A feed line stays on screen for several seconds, so it is seen on many
samples. One sighting is never enough: a line must be read consistently on
``confirm_sightings`` samples before it is believed, and a believed line is
suppressed for ``dedup_window_s`` so the same entry cannot emit twice.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .ocr import fuzzy_pick
from .profile import Profile

# the slot digit is often read as I or L on HUD fonts
ACTOR_RE = re.compile(r"([A-Z][A-Z0-9]{1,6}?)\s*#?\s*([1-4IL])(?![0-9])")
_SLOT_FIX = {"I": "1", "L": "1"}


@dataclass
class FeedEvent:
    t: float
    kind: str                       # "knock" | "kill"
    victim_team: Optional[str]
    victim_slot: Optional[int]
    killer_team: Optional[str]
    killer_slot: Optional[int]
    confidence: float
    raw: str = ""
    sightings: int = 1


def _norm(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def _actor_by_name(text: str, prof: Profile, prefer: str
                   ) -> Tuple[Optional[str], Optional[int], float]:
    """Resolve a TEAMxPLAYER name against the roster.

    A feed that names players, which is what most overlays show, carries no
    slot number. The roster's order supplies one, so a named line can still be
    matched against the pip the panel saw go down.
    """
    import difflib
    t = _norm(text)
    if not prof.roster or len(t) < 3:
        return None, None, 0.0
    best = (0.0, None, None, -1)
    for team, players in prof.roster.items():
        for i, player in enumerate(players):
            cand = _norm(f"{team}X{player}")
            if not cand:
                continue
            # find the best window of the text that looks like this label
            for start in range(0, max(1, len(t) - len(cand) + 2)):
                window = t[start:start + len(cand)]
                if len(window) < 3:
                    continue
                score = difflib.SequenceMatcher(None, window, cand).ratio()
                if score > best[0]:
                    best = (score, team, i + 1, start)
    score, team, slot, start = best
    if team is None or score < 0.75:
        return None, None, score
    return team, slot, score


def _actor(text: str, roster: List[str], prefer: str) -> Tuple[Optional[str], Optional[int], float]:
    matches = list(ACTOR_RE.finditer(text))
    if not matches:
        return None, None, 0.0
    m = matches[-1] if prefer == "last" else matches[0]
    tag, raw_slot = m.group(1), m.group(2)
    if raw_slot in _SLOT_FIX and roster:
        # 'I' and 'L' are here because OCR renders a slot '1' as either. But
        # they are also the last letter of real tags: SOUL, GODL. Taking the
        # letter as a slot invents 'slot 1' for every one of that team's feed
        # lines, at the tag's own confidence. The roster decides which it is:
        # if the tag reads better with the letter attached, it is part of the
        # name and this line simply carries no slot.
        joined, j_score = fuzzy_pick(tag + raw_slot, roster)
        alone, a_score = fuzzy_pick(tag, roster)
        if joined and j_score >= max(a_score, 0.6):
            # an unambiguous digit after the whole tag is still a slot:
            # 'SOUL#1' names SOUL slot 1, 'SOULXVIPER' names no slot at all.
            m2 = re.search(re.escape(tag + raw_slot) + r"\s*#?\s*([1-4])(?![0-9])", text)
            return joined, (int(m2.group(1)) if m2 else None), j_score
    slot = int(_SLOT_FIX.get(raw_slot, raw_slot))
    if roster:
        pick, score = fuzzy_pick(tag, roster)
        if pick and score >= 0.6:
            return pick, slot, score
        return None, None, 0.0
    return tag, slot, 0.7


def parse_feed_line(text: str, line_conf: float, prof: Profile) -> Optional[FeedEvent]:
    """Turn one OCR'd feed row into a knock/kill with killer attribution."""
    t = re.sub(r"\s+", " ", (text or "").upper()).strip()
    if len(t) < 5:
        return None
    kind = None
    kw_used = None
    for k, words in prof.feed_keywords.items():
        for w in sorted(words, key=len, reverse=True):
            if w in t:
                kind, kw_used = k, w
                break
        if kind:
            break
    if kind is None:
        return None
    left, _, right = t.partition(kw_used)
    ktag, kslot, kscore = _actor(left, prof.team_tags, "last")
    vtag, vslot, vscore = _actor(right, prof.team_tags, "first")
    if prof.roster:
        # a feed that names players beats one parsed as tag plus slot number
        nk, nks, nkscore = _actor_by_name(left, prof, "last")
        nv, nvs, nvscore = _actor_by_name(right, prof, "first")
        # a tie goes to the read that carries a slot: a tag-only read scores
        # on the tag alone, so it ties easily while saying strictly less.
        if nvscore > vscore or (nvscore >= vscore and nvs is not None and vslot is None):
            vtag, vslot, vscore = nv, nvs, nvscore
        if nkscore > kscore or (nkscore >= kscore and nks is not None and kslot is None):
            ktag, kslot, kscore = nk, nks, nkscore
    if vtag is None:
        return None   # no identifiable victim: not an event
    conf = min(1.0, 0.25 + 0.35 * line_conf + 0.4 * vscore)
    if ktag is None:
        conf *= 0.9   # killer unknown, victim still solid
    return FeedEvent(0.0, kind, vtag, vslot, ktag, kslot, conf, raw=t)


@dataclass
class FeedTracker:
    prof: Profile
    confirm_sightings: int = 2
    _pending: Dict[tuple, FeedEvent] = field(default_factory=dict)
    _emitted: Dict[tuple, float] = field(default_factory=dict)

    def key(self, ev: FeedEvent) -> tuple:
        # The victim identifies the event. The killer must NOT be part of the
        # key: one flickering character in the killer tag would otherwise make
        # a second key, and the same feed entry would be emitted twice.
        return (ev.kind, ev.victim_team, ev.victim_slot)

    def observe(self, t: float, lines: List[Tuple[str, float]]) -> List[FeedEvent]:
        out: List[FeedEvent] = []
        seen_now = set()
        for text, conf in lines:
            ev = parse_feed_line(text, conf, self.prof)
            if ev is None:
                continue
            k = self.key(ev)
            seen_now.add(k)
            last = self._emitted.get(k)
            if last is not None and (t - last) < self.prof.dedup_window_s:
                continue
            cur = self._pending.get(k)
            if cur is None:
                ev.t = t
                self._pending[k] = ev
                cur = ev
            else:
                cur.sightings += 1
                if ev.confidence > cur.confidence or (
                        cur.killer_team is None and ev.killer_team is not None):
                    # keep the most confident sighting's killer attribution
                    cur.killer_team, cur.killer_slot = ev.killer_team, ev.killer_slot
                    cur.raw = ev.raw
                cur.confidence = max(cur.confidence, ev.confidence)
            if cur.sightings >= self.confirm_sightings:
                cur.confidence = min(1.0, cur.confidence + 0.05 * (cur.sightings - 1))
                self._emitted[k] = cur.t
                out.append(cur)
                del self._pending[k]
        # a one-sighting candidate that vanished was OCR noise
        for k in [k for k in self._pending if k not in seen_now]:
            if self._pending[k].sightings < self.confirm_sightings:
                del self._pending[k]
        return out
