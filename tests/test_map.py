"""Unit tests for phase 3 (map stream): geometry, identity, zones, behaviour."""
from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest

from bgmi_scrim.map_dots import (ColourBook, Dot, _crowded, assign_teams_by_colour,
                                 detect_dots, match_label, match_team)
from bgmi_scrim.map_movement import (DEFAULTS, Phase, Sample, ZoneSample, edge_fraction,
                                     landing, phases, segments, zone_entry)
from bgmi_scrim.map_profile import MapProfile
from bgmi_scrim.map_track import Detection, Tracker
from bgmi_scrim.map_world import WorldMap, bearing_deg, compass, spread_m
from bgmi_scrim.map_zones import fit_circle_ransac


def _prof(**kw):
    base = {"map_area": [100, 50, 800, 800], "map": {"name": "ERANGEL", "world_size_m": 8000},
            "roster": {"TX": ["ALFA", "BRAVO"], "SOUL": ["VIPER"], "OR": ["ONE"]}}
    base.update(kw)
    return MapProfile.from_dict(base)


# ---------------- world ----------------
def test_pixels_to_metres_and_back():
    w = WorldMap((100, 50, 800, 800), 8000.0, "ERANGEL")
    assert w.to_world(100, 50) == (0.0, 0.0)
    assert w.to_world(900, 850) == (8000.0, 8000.0)
    x, y = w.to_pixels(4000, 4000)
    assert (round(x), round(y)) == (500, 450)
    assert w.metres_per_pixel() == 10.0
    assert w.grid_ref(10, 10) == "A1" and w.grid_ref(7990, 7990) == "H8"


def test_place_falls_back_to_a_grid_reference():
    from bgmi_scrim.map_world import Poi
    w = WorldMap((0, 0, 800, 800), 8000.0, "ERANGEL", [Poi("POCHINKI", 4000, 4000, 250)])
    assert w.place(4100, 4050) == "POCHINKI"
    assert w.place(1000, 7000) == "B8"       # outside every POI radius


def test_bearing_and_spread():
    assert compass(bearing_deg((0, 0), (0, -100))) == "N"
    assert compass(bearing_deg((0, 0), (100, 0))) == "E"
    assert spread_m([(0, 0), (300, 400)]) == pytest.approx(500.0)
    assert spread_m([(1, 1)]) is None


# ---------------- labels ----------------
@pytest.mark.parametrize("text,team,player", [
    ("TXXALFA", "TX", "ALFA"),
    ("TXxBRAVO", "TX", "BRAVO"),
    ("S0ULXVIPER", "SOUL", "VIPER"),
])
def test_label_resolves_against_the_roster(text, team, player):
    t, p, score = match_label(text, _prof())
    assert (t, p) == (team, player) and score >= 0.6


def test_a_name_that_is_on_no_roster_is_not_invented():
    t, p, _ = match_label("ZZZZXNOBODY", _prof())
    assert t is None and p is None


def test_team_survives_a_crowded_label_where_the_player_cannot():
    crowded = "TXXALFATXXBRAVOTXXCHARL"
    team, score = match_team(crowded, _prof())
    assert team == "TX" and score >= 0.66
    # a short tag must not match on a bare substring
    assert match_team("CHARLIEBRAVO", _prof())[0] != "OR"


def test_crowding_detects_a_neighbour_inside_the_label_box():
    a = Dot(px=100, py=100, area=60, radius=4, colour=(0, 200, 0))
    b = Dot(px=140, py=92, area=60, radius=4, colour=(0, 200, 0))
    assert _crowded(a, [a, b], (7, -16, 95, 15))
    far = Dot(px=400, py=400, area=60, radius=4, colour=(0, 200, 0))
    assert not _crowded(a, [a, far], (7, -16, 95, 15))


# ---------------- colour book ----------------
def test_colour_book_vetoes_a_name_on_the_wrong_colour():
    book = ColourBook(tol_deg=6.0, min_observations=2)
    for _ in range(3):
        book.observe("TX", (60, 220, 60))        # green
    assert book.agrees("TX", (60, 220, 60)) is True
    assert book.agrees("TX", (220, 60, 60)) is False
    assert book.agrees("SOUL", (60, 220, 60)) is None   # nothing learned yet


def test_colour_alone_does_not_name_a_team_when_hues_are_close():
    book = ColourBook(tol_deg=6.0, min_observations=2)
    for _ in range(3):
        book.observe("TX", (60, 220, 60))
        book.observe("SOUL", (60, 220, 75))      # nearly the same hue
    team, conf = book.team_from_colour((60, 220, 66))
    assert team is None                            # ambiguous: no identification


# ---------------- dots ----------------
def test_dots_are_found_and_a_merged_pair_is_reported_as_a_cluster():
    prof = _prof(dots={"min_area": 20, "max_area": 180, "sat_min": 90, "val_min": 90,
                       "single_area": 78.0, "cluster_ratio": 1.6, "max_cluster": 4,
                       "dedup_px": 4.0, "min_circularity": 0.45})
    img = np.full((900, 1000, 3), 40, np.uint8)
    cv2.circle(img, (300, 300), 5, (60, 220, 60), -1)      # one player
    cv2.circle(img, (500, 500), 5, (60, 220, 60), -1)      # two players merged
    cv2.circle(img, (506, 500), 5, (60, 220, 60), -1)
    cv2.circle(img, (700, 700), 120, (235, 150, 40), 3)    # a zone ring must not count
    dots = detect_dots(img, prof)
    assert len(dots) == 2
    assert sorted(d.cluster_size for d in dots) == [1, 2]


# ---------------- tracking ----------------
def test_track_keeps_its_identity_between_label_reads():
    tk = Tracker(max_speed_mps=50, max_miss_s=4, gate_floor_m=30)
    tk.update(0.0, [Detection(100, 100, team="TX", player="ALFA", label_conf=0.9, team_conf=0.9)])
    tk.update(0.5, [Detection(110, 100)])
    live = tk.update(1.0, [Detection(120, 100)])
    assert live[0].identity()[:2] == ("TX", "ALFA")


def test_two_live_tracks_cannot_be_the_same_player():
    tk = Tracker(max_speed_mps=50, max_miss_s=4, gate_floor_m=30)
    live = tk.update(0.0, [
        Detection(100, 100, team="TX", player="ALFA", label_conf=0.95, team_conf=0.95),
        Detection(900, 900, team="TX", player="ALFA", label_conf=0.3, team_conf=0.3)])
    ids = tk.resolve_frame(live)
    named = [v for v in ids.values() if v[0] is not None]
    assert len(named) == 1 and tk.id_switch_guard == 1


def test_a_dot_further_than_the_speed_gate_starts_a_new_track():
    tk = Tracker(max_speed_mps=10, max_miss_s=4, gate_floor_m=20)
    tk.update(0.0, [Detection(0, 0)])
    live = tk.update(1.0, [Detection(5000, 5000)])
    assert len(live) == 1 and live[0].id == 2


# ---------------- zone circles ----------------
def test_ransac_finds_the_circle_through_scattered_text_pixels():
    rng = np.random.default_rng(3)
    ang = rng.uniform(0, 2 * math.pi, 600)
    ring = np.stack([300 + 200 * np.cos(ang), 250 + 200 * np.sin(ang)], axis=1)
    noise = rng.uniform(0, 600, (300, 2))          # map text, icons, dots
    pts = np.vstack([ring, noise])
    cx, cy, r, inl = fit_circle_ransac(pts, 20, 500)
    assert abs(cx - 300) < 4 and abs(cy - 250) < 4 and abs(r - 200) < 4
    assert inl > 500


# ---------------- behaviour ----------------
def test_segments_split_standing_still_from_moving():
    s = [Sample(t * 0.5, 100.0, 100.0) for t in range(80)]       # 40 s still
    s += [Sample(40 + t * 0.5, 100 + t * 6.0, 100.0) for t in range(60)]  # then moving
    segs = segments(s, DEFAULTS)
    kinds = [x.kind for x in segs]
    assert "loot" in kinds and "rotation" in kinds
    rot = next(x for x in segs if x.kind == "rotation")
    assert rot.distance_m > 250 and rot.displacement_m > 250 and rot.mode == "vehicle"


def test_jitter_in_one_place_is_not_a_rotation():
    """Path length alone would call this 480 m of travel; it goes nowhere."""
    s = [Sample(t * 0.5, 100 + (t % 3) * 8.0, 100.0) for t in range(60)]
    segs = segments(s, DEFAULTS)
    assert all(x.kind != "rotation" for x in segs)


def test_phases_come_from_the_safe_circle_and_closes_from_the_blue():
    z = [ZoneSample(t, "safe", 4000, 4000, 3000) for t in (0, 20, 40)]
    z += [ZoneSample(t, "safe", 3500, 3500, 1500) for t in (60, 62, 64)]
    z += [ZoneSample(0, "blue", 4000, 4000, 3000),
          ZoneSample(95, "blue", 3500, 3500, 1500)]
    ps = phases(z)
    assert [p.number for p in ps] == [1, 2]
    assert ps[1].announced_t == 60 and ps[1].closed_t == 95


def test_one_bad_circle_fit_does_not_invent_a_phase():
    z = [ZoneSample(t, "safe", 4000, 4000, 3000) for t in range(0, 20, 2)]
    z.insert(5, ZoneSample(9, "safe", 1200, 7000, 900))      # a single wild fit
    z.sort(key=lambda s: s.t)
    assert [p.number for p in phases(z)] == [1]


def test_the_zone_stabiliser_holds_a_jump_back_until_it_repeats():
    from bgmi_scrim.map_zones import Circle, ZoneStabiliser
    st = ZoneStabiliser(confirm=2)
    for _ in range(3):
        assert st.accept("safe", Circle(4000, 4000, 2000, 0.9)) is not None
    assert st.accept("safe", Circle(1000, 7000, 400, 0.9)) is None    # held back
    assert st.accept("safe", Circle(1005, 7002, 402, 0.9)) is not None  # repeated: real
    assert st.rejected == 1


def test_zone_entry_reports_the_margin_to_the_close():
    ph = Phase(2, 60.0, 3500, 3500, 1500, closed_t=120.0)
    s = [Sample(60, 8000, 8000), Sample(80, 4600, 4600), Sample(100, 3600, 3600)]
    t_in, margin = zone_entry(s, ph)
    assert t_in == 100 and margin == 20.0
    assert zone_entry([Sample(60, 8000, 8000)], ph) is None


def test_edge_fraction_counts_time_on_the_rim():
    ph = Phase(1, 0.0, 4000, 4000, 1000.0)
    centre = [Sample(t, 4000, 4000) for t in range(0, 10)]
    rim = [Sample(t, 4950, 4000) for t in range(10, 20)]
    f = edge_fraction(centre + rim, [ph], 0.25)
    assert 0.4 < f < 0.6


# ---------------- end to end on rendered frames ----------------
def test_pipeline_reads_a_rendered_map_frame():
    from bgmi_scrim.map_synth import MapSim, render_map_frame
    from bgmi_scrim.map_world import WorldMap as WM
    prof = MapProfile.load("profiles/synthetic_map_1080p.json")
    sim = MapSim(duration=200, seed=5).build()
    frame = render_map_frame(sim, 150.0, 1920, 1080, (422, 43, 972, 972))
    dots = detect_dots(frame, prof)
    assert len(dots) > 10
    w = WM(tuple(prof.map_area), 8000.0)
    truth = [sim.position(t, p, 150.0) for t in sim.pois and ["TX"] for p in ["ALFA"]]
    # every dot should sit on some living player
    alive = [sim.position(tm, pl, 150.0)
             for tm in __import__("bgmi_scrim.map_synth", fromlist=["TEAMS"]).TEAMS[:25]
             for pl in ["ALFA", "BRAVO", "CHARL", "DELTA"]]
    alive = [a for a in alive if a]
    for d in dots:
        wx, wy = w.to_world(d.px, d.py)
        assert min(math.dist((wx, wy), a) for a in alive) < 60


def test_landing_is_the_end_of_the_descent_not_the_wait_in_the_plane():
    """A dot sitting still in the aircraft is slow; only the descent tells you
    where a squad actually came down."""
    from bgmi_scrim.map_movement import landing
    plane = [Sample(t * 0.5, 100.0, 0.0) for t in range(20)]            # waiting
    descent = [Sample(10 + t * 0.5, 100.0 + t * 60.0, 0.0) for t in range(20)]  # 120 m/s
    ground = [Sample(20 + t * 0.5, 1240.0 + t * 1.0, 0.0) for t in range(20)]
    land, saw = landing(plane + descent + ground, DEFAULTS)
    assert saw is True
    assert land.t >= 19.0 and land.x > 1000
    # a stream that starts after the drop says so instead of guessing
    land2, saw2 = landing(ground, DEFAULTS)
    assert saw2 is False and land2.t == ground[0].t


def test_a_tracking_hole_is_not_movement():
    a = [Sample(t * 0.5, 100.0 + t * 8.0, 100.0) for t in range(20)]     # moving
    b = [Sample(300 + t * 0.5, 5000.0 + t * 8.0, 100.0) for t in range(20)]
    segs = segments(a + b, DEFAULTS)
    assert all(x.t1 - x.t0 < 60 for x in segs), [(x.kind, x.t0, x.t1) for x in segs]
    assert all(x.distance_m < 1000 for x in segs)


def test_a_rotation_through_two_circles_is_split_at_the_announcements():
    """A squad that keeps driving has made the rotation for each circle, and a
    coach reads them per phase."""
    from bgmi_scrim.map_movement import split_at
    s = [Sample(t * 1.0, 1000 + t * 14.0, 2000.0) for t in range(60)]
    seg = segments(s, DEFAULTS)[0]
    assert seg.kind == "rotation" and seg.t1 - seg.t0 > 50
    parts = split_at(seg, s, [20.0, 40.0], DEFAULTS)
    assert len(parts) == 3
    assert [round(p.t0) for p in parts] == [0, 20, 40]
    assert sum(p.distance_m for p in parts) == pytest.approx(seg.distance_m, rel=0.05)
    assert all(p.mode == "vehicle" for p in parts)
    # a movement inside one phase is left alone
    assert len(split_at(seg, s, [200.0], DEFAULTS)) == 1


def test_a_tiny_final_circle_is_still_fitted():
    """The last circle of a match is a few pixels across and honestly holds only
    a few dozen points; a flat minimum threw it away and lost the final phase."""
    from bgmi_scrim.map_profile import MapProfile
    from bgmi_scrim.map_world import WorldMap
    from bgmi_scrim.map_zones import read_circles
    prof = MapProfile.from_dict({
        "map_area": [0, 0, 650, 650], "map": {"name": "ERANGEL", "world_size_m": 8000},
        "circles": {"safe": {"hsv_lo": [0, 0, 215], "hsv_hi": [179, 45, 255],
                             "min_radius_frac": 0.01, "max_radius_frac": 0.9,
                             "min_points": 150, "tol_px": 2.7}}})
    frame = np.full((650, 650, 3), 30, np.uint8)
    cv2.circle(frame, (300, 320), 8, (240, 240, 240), 2, cv2.LINE_AA)   # ~50 points
    world = WorldMap((0, 0, 650, 650), 8000.0, "ERANGEL")
    c = read_circles(frame, prof, world)["safe"]
    assert c is not None, "the final circle was rejected for holding few points"
    assert abs(c.r - 8 * world.metres_per_pixel()) < 40


def test_a_circle_that_grows_is_not_a_new_phase():
    """A fit that jumped to a 2822 m circle between the 115 m and 66 m ones
    invented a ninth phase in a match that had eight."""
    z = [ZoneSample(t, "safe", 3860, 4938, 115) for t in (0, 10, 20)]
    z += [ZoneSample(t, "safe", 5795, 2836, 2822) for t in (30, 32)]   # a bad fit, twice
    z += [ZoneSample(t, "safe", 3869, 4938, 66) for t in (40, 42, 44)]
    got = phases(z)
    assert [round(p.r) for p in got] == [115, 66]


def test_the_stabiliser_rejects_a_fit_that_grew():
    from bgmi_scrim.map_zones import Circle, ZoneStabiliser
    st = ZoneStabiliser(confirm=2)
    for _ in range(3):
        st.accept("safe", Circle(3860, 4938, 115, 0.9))
    assert st.accept("safe", Circle(5795, 2836, 2822, 0.9)) is None
    assert st.accept("safe", Circle(5795, 2836, 2822, 0.9)) is None   # even twice
    assert st.accept("safe", Circle(3869, 4938, 66, 0.9)) is not None


def test_a_short_tag_does_not_absorb_a_longer_one():
    """OR took 1564 samples that belonged to other teams once 720p garbled the
    labels: a two-letter tag fits inside half the roster."""
    prof = MapProfile.from_dict({"roster": {"OR": ["ALFA"], "ORCA": ["ALFA"],
                                            "TX": ["BRAVO"]}})
    # a clean label still resolves
    assert match_label("ORCAXALFA", prof)[:2] == ("ORCA", "ALFA")
    assert match_label("ORXALFA", prof)[:2] == ("OR", "ALFA")
    # a garbled one that fits both says nothing rather than picking the short tag
    team, player, _score = match_label("0RCAXALFA", prof)
    assert team in (None, "ORCA")
    assert match_team("ORCAXALFA", prof)[0] == "ORCA"


def test_a_label_that_fits_two_teams_equally_is_not_named():
    prof = MapProfile.from_dict({"roster": {"GB": ["ALFA"], "GD": ["ALFA"]}})
    team, player, _ = match_label("GXXALFA", prof)
    assert team is None
