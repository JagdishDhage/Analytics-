"""Fights and eliminations from positions, with no kill feed to check against.

The synthetic map footage cannot test this end to end: in a rendered clip the
closest two teams ever come is 146 m by centroid and 523 m by nearest player
pair, so no contact is ever opened. These tests therefore drive the logic with
constructed paths. That proves the derivation, not the footage -- see
docs/corpus.md.
"""
from __future__ import annotations

import math

from bgmi_scrim.contacts import (DEFAULTS, Path, contacts, losses, paths_from_samples,
                                 placements, team_eliminations)


def _samples(team, player, pts, t0=0.0, step=0.5, ident=1.0):
    return [{"type": "sample", "t": t0 + i * step, "team": team, "player": player,
             "x_m": x, "y_m": y, "identity_confidence": ident}
            for i, (x, y) in enumerate(pts)]


def _hold(team, player, x, y, t0, t1, step=0.5, ident=1.0):
    n = int((t1 - t0) / step) + 1
    return _samples(team, player, [(x, y)] * n, t0=t0, step=step, ident=ident)


def test_a_team_walking_into_another_is_one_contact():
    # B sits still; A drives in from 2 km away, arrives at 100 s, stays
    end = 200.0
    approach = [(2000.0 - 20.0 * (i * 0.5), 0.0) for i in range(201)]   # 0..100 s
    rows = _samples("A", "ALFA", approach)
    rows += _hold("A", "ALFA", 0.0, 0.0, 100.5, end)
    rows += _hold("B", "BRAVO", 50.0, 0.0, 0.0, end)
    cs = contacts(paths_from_samples(rows), end)
    assert len(cs) == 1, [(c.t0, c.t1, c.min_dist_m) for c in cs]
    c = cs[0]
    assert c.min_dist_m <= DEFAULTS["contact_m"]
    assert {c.team_a, c.team_b} == {"A", "B"}


def test_the_side_that_closed_the_ground_is_the_initiator():
    end = 200.0
    approach = [(2000.0 - 20.0 * (i * 0.5), 0.0) for i in range(201)]
    rows = _samples("A", "ALFA", approach)
    rows += _hold("A", "ALFA", 0.0, 0.0, 100.5, end)
    rows += _hold("B", "BRAVO", 50.0, 0.0, 0.0, end)
    c = contacts(paths_from_samples(rows), end)[0]
    assert c.initiator == "A", (c.closed_a, c.closed_b, c.initiator)
    assert c.mode_a == "vehicle"          # 20 m/s on the way in


def test_two_teams_passing_at_distance_is_not_a_contact():
    end = 120.0
    rows = _hold("A", "ALFA", 0.0, 0.0, 0.0, end)
    rows += _hold("B", "BRAVO", 900.0, 0.0, 0.0, end)
    assert contacts(paths_from_samples(rows), end) == []


def test_losses_inside_a_contact_are_counted_to_their_side():
    end = 200.0
    # both teams sit 60 m apart from 0 s; A's second player stops at 100 s
    rows = _hold("A", "ALFA", 0.0, 0.0, 0.0, end)
    rows += _hold("A", "BRAVO", 10.0, 0.0, 0.0, 100.0)
    rows += _hold("B", "CHARL", 60.0, 0.0, 0.0, end)
    ps = paths_from_samples(rows)
    cs = contacts(ps, end)
    assert len(cs) == 1
    c = cs[0]
    a_losses = c.losses_a if c.team_a == "A" else c.losses_b
    b_losses = c.losses_b if c.team_a == "A" else c.losses_a
    assert (a_losses, b_losses) == (1, 0)
    assert c.outcome_for("A") == "lost"
    assert c.outcome_for("B") == "won"


def test_a_dot_that_lasts_to_the_end_is_not_a_loss():
    end = 100.0
    rows = _hold("A", "ALFA", 0.0, 0.0, 0.0, end)
    assert losses(paths_from_samples(rows), end) == []


def test_a_dot_that_stops_early_is_a_loss():
    end = 300.0
    rows = _hold("A", "ALFA", 0.0, 0.0, 0.0, 100.0)
    ls = losses(paths_from_samples(rows), end)
    assert len(ls) == 1 and ls[0].player == "ALFA"
    assert abs(ls[0].t - 100.0) < 1.0
    assert 0.0 < ls[0].confidence <= 0.9


def test_placement_is_the_reverse_of_elimination_order():
    end = 400.0
    rows = _hold("DEAD1", "A", 0.0, 0.0, 0.0, 100.0)
    rows += _hold("DEAD2", "A", 500.0, 0.0, 0.0, 200.0)
    rows += _hold("DEAD3", "A", 1000.0, 0.0, 0.0, 300.0)
    rows += _hold("ALIVE", "A", 1500.0, 0.0, 0.0, end)
    ps = paths_from_samples(rows)
    el = team_eliminations(ps, end)
    assert el["ALIVE"] is None
    pl = placements(el)
    assert pl["ALIVE"] == 1
    assert pl["DEAD3"] == 2 and pl["DEAD2"] == 3 and pl["DEAD1"] == 4


def test_teams_still_alive_at_the_end_share_the_top_rank():
    """The stream ending is not a result. Two teams still up cannot be ordered."""
    end = 200.0
    rows = _hold("X", "A", 0.0, 0.0, 0.0, end)
    rows += _hold("Y", "A", 900.0, 0.0, 0.0, end)
    rows += _hold("Z", "A", 1800.0, 0.0, 0.0, 50.0)
    pl = placements(team_eliminations(paths_from_samples(rows), end))
    assert pl["X"] == pl["Y"] == 1
    assert pl["Z"] == 3        # two above it, so third


def test_paths_group_by_player_not_by_track_id():
    """The tracker opens a new track every time it loses a dot."""
    rows = _hold("A", "ALFA", 0.0, 0.0, 0.0, 20.0)
    for r in rows:
        r["track"] = 1
    more = _hold("A", "ALFA", 100.0, 0.0, 40.0, 60.0)
    for r in more:
        r["track"] = 77
    ps = paths_from_samples(rows + more)
    assert len(ps) == 1, [(p.team, p.player, len(p.t)) for p in ps]
    assert ps[0].t0 == 0.0 and ps[0].t1 == 60.0


def test_a_wide_gap_is_not_interpolated_across():
    """A dot seen at 10 s and again at 400 s was not in between."""
    rows = _hold("A", "ALFA", 0.0, 0.0, 0.0, 10.0)
    rows += _hold("A", "ALFA", 5000.0, 0.0, 400.0, 410.0)
    p = paths_from_samples(rows)[0]
    assert p.at(200.0, 3.0) is None
