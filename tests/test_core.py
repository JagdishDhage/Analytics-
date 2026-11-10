"""Unit tests for the parts that do not need a video: confirmation, parsing,
fusion, scoring and the phase-2 report."""
from __future__ import annotations

import json

import numpy as np
import pytest

from bgmi_scrim.events import Event
from bgmi_scrim.fuse import Fuser, PanelEvent
from bgmi_scrim.killfeed import FeedTracker, parse_feed_line
from bgmi_scrim.preprocess import colour_fraction, prepare_roi, roi_is_blank
from bgmi_scrim.profile import Profile
from bgmi_scrim.statemachine import UNKNOWN, ConfirmedField


def _prof(**kw):
    base = {"team_tags": ["TX", "GB", "SOUL"], "resolution": [1920, 1080]}
    base.update(kw)
    return Profile.from_dict(base)


# ---------------- profile ----------------
def test_profile_roundtrip_and_scaling(tmp_path):
    p = Profile.load("profiles/synthetic_1080p.json")
    out = tmp_path / "p.json"
    p.save(out)
    q = Profile.load(out)
    assert q.to_dict() == p.to_dict()
    r = p.for_frame_size(1280, 720)
    x, y, w, h = r.rois["teams_alive"].box
    bx, by, bw, bh = p.rois["teams_alive"].box
    assert abs(x / bx - 2 / 3) < 0.01 and abs(w / bw - 2 / 3) < 0.05
    assert r.rois["team_panel"].meta["row_pitch"] == round(
        p.rois["team_panel"].meta["row_pitch"] * 720 / 1080)


# ---------------- state machine ----------------
def test_confirmation_absorbs_single_frame_flicker():
    f = ConfirmedField("teams_alive", confirm_n=3, threshold=0.6, event_type="teams_alive")
    emitted = []
    for i, v in enumerate([25, 25, 25, 11, 25, 25, 25, 24, 24, 24]):
        ev = f.observe(i * 0.5, i, v, 0.9)
        if ev:
            emitted.append((ev.t, ev.value))
    assert emitted == [(0.0, 25), (3.5, 24)]


def test_low_confidence_becomes_unknown_not_a_guess():
    f = ConfirmedField("zone_phase", confirm_n=2, threshold=0.6, event_type="zone_phase")
    assert f.observe(0.0, 0, 3, 0.9) is None
    ev = f.observe(0.5, 1, 3, 0.9)
    assert ev.value == 3
    f.observe(1.0, 2, 7, 0.1)
    ev = f.observe(1.5, 3, 9, 0.2)
    assert ev is not None and ev.state == "unknown" and ev.value is None


def test_counter_never_jumps_up_inside_a_match():
    f = ConfirmedField("teams_alive", confirm_n=2, threshold=0.6, monotonic="down",
                       reset_at=20, event_type="teams_alive")
    f.observe(0.0, 0, 10, 0.9)
    f.observe(0.5, 1, 10, 0.9)
    assert f.current == 10
    f.observe(1.0, 2, 18, 0.9)
    assert f.observe(1.5, 3, 18, 0.9) is None      # rejected: counters only fall
    assert f.rejected == 1
    f.observe(2.0, 4, 25, 0.9)
    ev = f.observe(2.5, 5, 25, 0.9)                 # reset: a new match started
    assert ev is not None and ev.value == 25


# ---------------- kill feed ----------------
@pytest.mark.parametrize("line,kind,victim,killer", [
    ("GB2 KNOCKED TX4", "knock", ("TX", 4), ("GB", 2)),
    ("GB2KNOCKEDTX4", "knock", ("TX", 4), ("GB", 2)),
    ("SOUL1 FINISHED GB3", "kill", ("GB", 3), ("SOUL", 1)),
    ("TXL FINISHED GB1", "kill", ("GB", 1), ("TX", 1)),       # slot read as L
])
def test_feed_parsing(line, kind, victim, killer):
    ev = parse_feed_line(line, 0.5, _prof())
    assert ev is not None
    assert (ev.kind, (ev.victim_team, ev.victim_slot), (ev.killer_team, ev.killer_slot)) == (
        kind, victim, killer)


def test_feed_line_without_a_known_victim_is_not_an_event():
    assert parse_feed_line("ZZ9 KNOCKED QQ7", 0.9, _prof()) is None
    assert parse_feed_line("random hud text", 0.9, _prof()) is None


def test_feed_needs_two_sightings_and_then_dedups():
    tr = FeedTracker(_prof(), confirm_sightings=2)
    assert tr.observe(1.0, [("GB2 KNOCKED TX4", 0.8)]) == []
    out = tr.observe(1.5, [("GB2 KNOCKED TX4", 0.8)])
    assert len(out) == 1 and out[0].t == 1.0
    assert tr.observe(2.0, [("GB2 KNOCKED TX4", 0.8)]) == []    # same entry, suppressed
    assert tr.observe(2.5, [("GB2 KNOCKED TX4", 0.8)]) == []
    again = tr.observe(20.0, [("GB2 KNOCKED TX4", 0.8)]) or tr.observe(
        20.5, [("GB2 KNOCKED TX4", 0.8)])
    assert len(again) == 1                                       # past the dedup window


# ---------------- fusion ----------------
def _feed_ev(t, kind="kill", victim=("TX", 2), killer=("GB", 1), conf=0.9):
    from bgmi_scrim.killfeed import FeedEvent
    return FeedEvent(t, kind, victim[0], victim[1], killer[0], killer[1], conf, sightings=2)


def test_panel_event_takes_killer_from_the_feed():
    fz = Fuser(fuse_window_s=6.0)
    fz.add_panel(PanelEvent(10.0, 300, "kill", "TX", 2, 0.9))
    fz.add_feed(_feed_ev(10.4))
    out = fz.flush(now=None)
    assert len(out) == 1
    assert out[0].killer_team == "GB" and out[0].source == "panel+feed"


def test_feed_only_event_is_dropped_when_the_panel_contradicts_it():
    fz = Fuser(fuse_window_s=6.0, corroborates=lambda *_: False)
    fz.add_feed(_feed_ev(10.0))
    assert fz.flush(now=None) == []
    assert len(fz.suppressed_feed) == 1


def test_feed_only_event_is_kept_when_the_panel_saw_something_happen():
    fz = Fuser(fuse_window_s=6.0, corroborates=lambda *_: True)
    fz.add_feed(_feed_ev(10.0))
    out = fz.flush(now=None)
    assert len(out) == 1 and out[0].source == "feed"


# ---------------- preprocessing ----------------
def test_blank_roi_is_gated_out_before_ocr():
    flat = np.full((40, 200, 3), 20, np.uint8)
    assert roi_is_blank(prepare_roi(flat, (0, 0, 200, 40), {"upscale": 2}))
    import cv2
    img = flat.copy()
    cv2.putText(img, "GB2 KNOCKED TX4", (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (235, 235, 235), 2, cv2.LINE_AA)
    assert not roi_is_blank(prepare_roi(img, (0, 0, 200, 40), {"upscale": 2}))


def test_colour_fraction_handles_hue_wrap():
    red = np.zeros((10, 10, 3), np.uint8)
    red[:, :] = (0, 0, 220)
    assert colour_fraction(red, (170, 120, 120), (10, 255, 255)) > 0.9


# ---------------- truth harness ----------------
def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    return str(p)


def test_truth_counts_misses_duplicates_and_phantoms(tmp_path):
    from bgmi_scrim.truth import score
    labels = [
        {"kind": "event", "t": 10.0, "type": "kill", "victim_team": "TX", "victim_slot": 1,
         "killer_team": "GB", "killer_slot": 2},
        {"kind": "event", "t": 20.0, "type": "kill", "victim_team": "GB", "victim_slot": 3},
        {"kind": "frame", "t": 10.0, "teams_alive": 20},
    ]
    events = [
        {"t": 10.2, "type": "kill", "victim_team": "TX", "victim_slot": 1,
         "killer_team": "GB", "killer_slot": 2, "confidence": 0.9, "state": "ok"},
        {"t": 10.6, "type": "kill", "victim_team": "TX", "victim_slot": 1,
         "confidence": 0.8, "state": "ok"},                       # duplicate
        {"t": 50.0, "type": "kill", "victim_team": "SOUL", "victim_slot": 2,
         "confidence": 0.8, "state": "ok"},                       # phantom
        {"t": 5.0, "type": "teams_alive", "value": 20, "confidence": 0.9, "state": "ok"},
    ]
    rep = score(_write(tmp_path, "l.jsonl", labels), _write(tmp_path, "e.jsonl", events))
    es = rep.events["kill"]
    assert (es.truth, es.predicted, es.matched) == (2, 3, 1)
    assert es.duplicates == 1 and es.phantoms == 1 and es.misses == 1
    assert es.killer_correct == 1
    assert rep.fields["teams_alive"].accuracy == 1.0
    kinds = {c["kind"] for c in rep.confusion}
    assert {"duplicate_event", "phantom_event", "missed_event"} <= kinds
    assert rep.gate()["passed"] is False


def test_truth_flags_a_miss_that_the_log_marked_unknown(tmp_path):
    from bgmi_scrim.truth import score
    labels = [{"kind": "frame", "t": 30.0, "teams_alive": 12}]
    events = [{"t": 10.0, "type": "teams_alive", "value": None, "confidence": 0.2,
               "state": "unknown"}]
    rep = score(_write(tmp_path, "l.jsonl", labels), _write(tmp_path, "e.jsonl", events))
    fs = rep.fields["teams_alive"]
    assert fs.wrong_flagged == 1 and fs.wrong_confident == 0
    assert rep.confusion[0]["flagged"] is True


# ---------------- phase 2 ----------------
def test_report_derives_only_from_the_log(tmp_path):
    from bgmi_scrim.report import build_report
    evs = [
        {"t": 0.0, "type": "match_start", "confidence": 0.9, "state": "ok", "match_index": 0},
        {"t": 1.0, "type": "map_name", "value": "ERANGEL", "confidence": 0.9, "state": "ok",
         "match_index": 0},
        {"t": 5.0, "type": "zone_phase", "value": 1, "confidence": 0.9, "state": "ok",
         "match_index": 0},
        {"t": 60.0, "type": "zone_phase", "value": 3, "confidence": 0.9, "state": "ok",
         "match_index": 0},
        {"t": 30.0, "type": "knock", "victim_team": "GB", "victim_slot": 1,
         "killer_team": "TX", "killer_slot": 2, "confidence": 0.9, "state": "ok",
         "match_index": 0},
        {"t": 33.0, "type": "kill", "victim_team": "GB", "victim_slot": 1,
         "killer_team": "TX", "killer_slot": 2, "confidence": 0.9, "state": "ok",
         "match_index": 0},
        {"t": 40.0, "type": "knock", "victim_team": "TX", "victim_slot": 1,
         "killer_team": "SOUL", "killer_slot": 3, "confidence": 0.9, "state": "ok",
         "match_index": 0},
        {"t": 44.0, "type": "kill", "victim_team": "TX", "victim_slot": 1,
         "killer_team": "SOUL", "killer_slot": 3, "confidence": 0.9, "state": "ok",
         "match_index": 0},
        {"t": 300.0, "type": "placement", "value": 4, "team": "TX", "confidence": 0.9,
         "state": "ok", "match_index": 0},
        {"t": 310.0, "type": "match_end", "confidence": 0.9, "state": "ok", "match_index": 0},
        # a low-confidence record that must not reach the report
        {"t": 100.0, "type": "kill", "victim_team": "GB", "victim_slot": 4,
         "killer_team": "TX", "killer_slot": 1, "confidence": 0.2, "state": "ok",
         "match_index": 0},
    ]
    md = build_report([_write(tmp_path, "e.jsonl", evs)], team="TX", block_name="test block")
    assert "M1 00:33" in md                      # our kill cited with match and timestamp
    assert "M1 00:44" in md                      # our death cited the same way
    assert "| M1 | ERANGEL | #4" in md
    assert "Not measurable yet" in md
    assert "1 point per kill" in md
    assert "M1 01:40" not in md                  # the low-confidence kill is excluded
    assert md.count("knock") >= 1
    # our kills = 1 (the 0.2-confidence one is excluded) -> 4 placement pts + 1 kill pt
    assert "| 1 | 1 | 5 | 5 |" in md


# ---------------- sampling schedule ----------------
def test_burst_window_raises_the_rate_around_a_change():
    from bgmi_scrim.sampler import VideoInfo, base_schedule, burst_schedule, merge_schedules
    info = VideoInfo("x", 30.0, 30 * 60, 1920, 1080)      # 60 s at 30 fps
    base = base_schedule(info, 2.0)
    assert base[:3] == [0, 15, 30] and len(base) == 120
    burst = burst_schedule(info, [10.0], burst_fps=5.0, window_s=3.0)
    assert burst[0] == int(8.5 * 30) and burst[-1] == pytest.approx(11.5 * 30, abs=6)
    assert all(b - a == 6 for a, b in zip(burst, burst[1:]))   # 5 fps at 30 fps source
    merged = merge_schedules(base, burst)
    assert merged == sorted(set(merged))
    in_window = [i for i in merged if 8.5 * 30 <= i <= 11.5 * 30]
    assert len(in_window) >= 15


# ---------------- panel row identity ----------------
def test_tag_resolver_assigns_each_team_to_one_row():
    from bgmi_scrim.analyze import PanelTagResolver
    r = PanelTagResolver(3, ["TX", "GB", "SOUL"])
    r.vote(0, "TX", 0.9); r.vote(0, "GB", 0.2)
    r.vote(1, "GB", 0.8); r.vote(1, "TX", 0.7)
    r.vote(2, "SOUL", 0.6)
    assert r.resolve() == {0: "TX", 1: "GB", 2: "SOUL"}
    assert r.tag(0) == "TX" and r.resolved(2)


def test_unresolved_row_is_named_not_guessed():
    from bgmi_scrim.analyze import PanelTagResolver
    r = PanelTagResolver(2, ["TX"])
    r.vote(0, "TX", 0.9)
    r.resolve()
    assert r.tag(1) == "ROW2" and not r.resolved(1)


def test_row_moves_only_after_two_decisive_epochs():
    from bgmi_scrim.analyze import PanelTagResolver
    r = PanelTagResolver(2, ["TX", "GB"])
    r.vote(0, "TX", 0.9); r.vote(1, "GB", 0.9)
    r.resolve()
    r.vote(0, "GB", 0.9); r.end_epoch()
    assert r.tag(0) == "TX"          # one disagreeing epoch is not enough
    r.vote(0, "GB", 0.9); r.end_epoch()
    assert r.tag(0) == "GB" and r.reorders == 1


# ---------------- transitions ----------------
def test_only_real_slot_transitions_become_events():
    from bgmi_scrim.analyze import Analyzer
    from bgmi_scrim.hud import ALIVE, DEAD, KNOCKED
    k = Analyzer._transition_kind
    assert k(ALIVE, KNOCKED) == "knock"
    assert k(KNOCKED, DEAD) == "kill"
    assert k(ALIVE, DEAD) == "kill"
    assert k(KNOCKED, ALIVE) is None     # a revive is not an event
    assert k(None, ALIVE) is None        # first sighting of a slot is not an event
    assert k(UNKNOWN, DEAD) is None
