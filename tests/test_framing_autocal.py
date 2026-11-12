"""Framing and auto-calibration: what the tool works out about a new capture."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from bgmi_scrim.autocal import (_modal_area, propose_counters, propose_kill_feed,
                                propose_map_area, propose_team_panel)
from bgmi_scrim.framing import detect_content_box, parse_box
from bgmi_scrim.profile import Profile


def _noise(h, w, seed=1):
    return np.random.default_rng(seed).integers(40, 220, (h, w, 3), dtype=np.uint8)


# ---------------- framing ----------------
def test_letterbox_and_pillarbox_are_found():
    f = np.zeros((1080, 1920, 3), np.uint8)
    f[27:27 + 1026, 48:48 + 1824] = _noise(1026, 1824)
    cb = detect_content_box([f])
    assert (cb.x, cb.y, cb.w, cb.h) == (48, 27, 1824, 1026)
    assert not cb.is_full_frame and cb.confidence > 0.6

    p = np.zeros((1080, 1920, 3), np.uint8)
    p[:, 240:240 + 1440] = _noise(1080, 1440, 2)
    cb2 = detect_content_box([p])
    assert (cb2.x, cb2.w) == (240, 1440) and cb2.y == 0


def test_a_full_frame_is_reported_as_one():
    cb = detect_content_box([_noise(1080, 1920, 3)])
    assert cb.is_full_frame and "whole 1920x1080" in cb.describe()


def test_a_dark_scene_does_not_look_like_a_bar():
    """One dark frame should not shrink the picture; the brightest frame wins."""
    dark = np.full((1080, 1920, 3), 5, np.uint8)
    dark[400:700, 400:700] = 200
    bright = _noise(1080, 1920, 4)
    assert detect_content_box([dark, bright]).is_full_frame


def test_parse_box():
    assert parse_box("48,27,1824,1026") == (48, 27, 1824, 1026)
    assert parse_box("auto") is None and parse_box("") is None
    with pytest.raises(ValueError):
        parse_box("1,2,3")


def test_a_profile_fits_into_the_content_box():
    p = Profile.load("profiles/synthetic_1080p.json")
    fitted = p.fitted(1920, 1080, (48, 27, 1824, 1026))
    native = p.rois["teams_alive"].box
    got = fitted.rois["teams_alive"].box
    # the box moves with the picture and shrinks with it
    assert got[0] == pytest.approx(48 + native[0] * 1824 / 1920, abs=2)
    assert got[1] == pytest.approx(27 + native[1] * 1026 / 1080, abs=2)
    assert p.fitted(1920, 1080, None).rois["teams_alive"].box == native


# ---------------- auto-calibration ----------------
def _panel_frame(rows=25, slots=4, pitch=34, x0=1722, y0=124, slot=18, gap=26):
    f = np.full((1080, 1920, 3), 20, np.uint8)
    for r in range(rows):
        for s in range(slots):
            x = x0 + s * gap
            y = y0 + r * pitch
            cv2.rectangle(f, (x, y), (x + slot, y + slot), (60, 200, 60), -1)
        cv2.putText(f, "TEAM", (x0 - 80, y0 + r * pitch + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return f


def test_the_team_list_is_found_from_its_regular_pips():
    p = propose_team_panel([_panel_frame()])
    panel = p.fields["team_panel"]
    assert panel["rows"] == 25 and panel["slots"] == 4
    assert panel["row_pitch"] == pytest.approx(34, abs=0.5)
    assert panel["slot_pitch"] == pytest.approx(26, abs=0.5)
    assert panel["box"][0] + panel["slot_x0"] == pytest.approx(1722, abs=3)
    assert p.confidence > 0.6


def test_nothing_is_proposed_when_there_is_no_panel():
    p = propose_team_panel([np.full((1080, 1920, 3), 20, np.uint8)])
    assert p.fields == {} and p.confidence == 0.0
    assert any("no regular stack" in n for n in p.notes)


def test_counters_are_told_apart_by_what_they_read():
    frames = []
    for players in (100, 88, 61, 42):
        f = np.full((1080, 1920, 3), 18, np.uint8)
        cv2.putText(f, "25", (1705, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (235, 235, 235),
                    2, cv2.LINE_AA)
        cv2.putText(f, str(players), (1825, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (235, 235, 235), 2, cv2.LINE_AA)
        frames.append(f)
    p = propose_counters(frames)
    assert "teams_alive" in p.fields and "players_alive" in p.fields
    tx = p.fields["teams_alive"]["box"][0]
    px = p.fields["players_alive"]["box"][0]
    assert abs(tx - 1701) < 12 and abs(px - 1821) < 12
    assert tx != px


def test_the_map_square_is_found_from_the_dots_and_the_ring():
    f = np.full((1080, 1920, 3), 30, np.uint8)
    rng = np.random.default_rng(7)
    for _ in range(120):
        x = int(rng.integers(520, 1290))
        y = int(rng.integers(140, 910))
        cv2.circle(f, (x, y), 5, (60, 220, 60), -1, cv2.LINE_AA)
    cv2.circle(f, (900, 520), 420, (235, 150, 40), 3, cv2.LINE_AA)   # zone ring
    p = propose_map_area([f, f])
    x, y, w, h = p.fields["map_area"]
    assert w == h                                  # a map is square
    assert x < 500 and x + w > 1290 and y < 150    # covers the dots and the ring
    assert 60 <= p.fields["dots"]["single_area"] <= 160


def test_one_dot_is_sized_by_the_commonest_blob_not_the_average():
    sizes = [118] * 40 + [120] * 30 + [236] * 12 + [355] * 6 + [20] * 4
    assert _modal_area(sizes) == pytest.approx(119, abs=12)
    assert _modal_area(sizes) < float(np.mean(sizes))
