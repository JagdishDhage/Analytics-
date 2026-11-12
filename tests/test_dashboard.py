"""The dashboard's data model: it must carry what the logs say and nothing else."""
from __future__ import annotations

import json

import pytest

from bgmi_scrim.dashboard import build_model, render_html


def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    return str(p)


def _match(tmp_path, name="e.jsonl"):
    evs = [
        {"t": 0.0, "type": "match_start", "confidence": 0.9, "state": "ok", "match_index": 0},
        {"t": 1.0, "type": "map_name", "value": "ERANGEL", "confidence": 0.9, "state": "ok"},
        {"t": 2.0, "type": "teams_alive", "value": 25, "confidence": 0.9, "state": "ok"},
        {"t": 40.0, "type": "teams_alive", "value": 20, "confidence": 0.9, "state": "ok"},
        {"t": 5.0, "type": "zone_phase", "value": 1, "confidence": 0.9, "state": "ok"},
        {"t": 60.0, "type": "zone_phase", "value": 2, "confidence": 0.9, "state": "ok"},
        {"t": 30.0, "type": "knock", "victim_team": "GB", "victim_slot": 1,
         "killer_team": "TX", "killer_slot": 2, "confidence": 0.9, "state": "ok"},
        {"t": 33.0, "type": "kill", "victim_team": "GB", "victim_slot": 1,
         "killer_team": "TX", "killer_slot": 2, "confidence": 0.9, "state": "ok"},
        {"t": 44.0, "type": "kill", "victim_team": "TX", "victim_slot": 1,
         "confidence": 0.9, "state": "ok"},
        {"t": 90.0, "type": "placement", "value": 4, "team": "TX", "confidence": 0.9,
         "state": "ok"},
        {"t": 95.0, "type": "match_end", "confidence": 0.9, "state": "ok"},
    ]
    return _write(tmp_path, name, evs)


def test_model_carries_results_events_and_series(tmp_path):
    m = build_model([_match(tmp_path)], team="TX", block_name="test")
    assert m["team"] == "TX" and len(m["matches"]) == 1
    one = m["matches"][0]
    assert one["map"] == "ERANGEL" and one["placement"] == 4
    assert one["kills"] == 1 and one["total"] == 4 + 1       # #4 = 4 pts, 1 kill
    assert one["teamsAlive"][0] == [2.0, 25]
    kinds = {(e["kind"], e["side"]) for e in one["events"]}
    assert ("kill", "gain") in kinds and ("kill", "loss") in kinds
    assert one["events"][0]["t"] < one["events"][-1]["t"]
    assert one["map3"] is None                                # no map stream supplied


def test_a_missing_placement_stays_missing(tmp_path):
    evs = [{"t": 0.0, "type": "match_start", "confidence": 0.9, "state": "ok"},
           {"t": 10.0, "type": "kill", "victim_team": "TX", "victim_slot": 1,
            "confidence": 0.9, "state": "ok"},
           {"t": 20.0, "type": "match_end", "confidence": 0.9, "state": "ok"}]
    m = build_model([_write(tmp_path, "x.jsonl", evs)], team="TX")
    one = m["matches"][0]
    assert one["placement"] is None and one["placementPts"] == 0
    assert one["map"] is None


def test_low_confidence_records_are_counted_not_shown(tmp_path):
    evs = [{"t": 0.0, "type": "match_start", "confidence": 0.9, "state": "ok"},
           {"t": 12.0, "type": "kill", "victim_team": "GB", "victim_slot": 2,
            "killer_team": "TX", "killer_slot": 1, "confidence": 0.2, "state": "ok"},
           {"t": 14.0, "type": "teams_alive", "value": None, "confidence": 0.1,
            "state": "unknown"},
           {"t": 20.0, "type": "match_end", "confidence": 0.9, "state": "ok"}]
    m = build_model([_write(tmp_path, "y.jsonl", evs)], team="TX")
    assert m["matches"][0]["kills"] == 0
    assert m["quality"]["unknown"] == 1 and m["quality"]["low_confidence"] == 1


def test_map_stream_adds_paths_phases_and_landings(tmp_path):
    tracks = [{"type": "map_info", "map": "ERANGEL", "world_size_m": 8000, "our_team": "TX"}]
    for i in range(10):
        tracks.append({"type": "sample", "t": i * 2.0, "track": 1, "team": "TX",
                       "player": "ALFA", "x_m": 1000 + i * 50, "y_m": 2000,
                       "confidence": 0.9})
        tracks.append({"type": "sample", "t": i * 2.0, "track": 2, "team": "GB",
                       "x_m": 5000, "y_m": 5000, "confidence": 0.9})
    mev = [{"type": "zone_phase", "t": 0.0, "phase": 1, "cx_m": 4000, "cy_m": 4000,
            "r_m": 3000, "closed_t": 20.0, "place": "E5"},
           {"type": "landing", "t": 4.0, "team": "TX", "unit": "team", "x_m": 1100,
            "y_m": 2000, "place": "B3"},
           {"type": "rotation", "t": 6.0, "t_end": 18.0, "team": "TX", "unit": "team",
            "from": "B3", "to": "C3", "distance_m": 400, "mode": "foot", "phase": 1,
            "after_announce_s": 6.0}]
    m = build_model([_match(tmp_path)], team="TX",
                    tracks_paths=[_write(tmp_path, "t.jsonl", tracks)],
                    map_events_paths=[_write(tmp_path, "me.jsonl", mev)])
    mp = m["matches"][0]["map3"]
    assert mp["world"] == 8000 and set(mp["paths"]) == {"TX", "GB"}
    assert mp["phases"][0]["r"] == 3000 and mp["landings"][0]["place"] == "B3"
    assert mp["rotations"][0]["mode"] == "foot"
    assert all(len(p) == 3 for p in mp["paths"]["TX"])


def test_rendered_page_is_self_contained(tmp_path):
    m = build_model([_match(tmp_path)], team="TX", block_name="test block")
    html = render_html(m)
    assert "/*%%MODEL%%*/null" not in html
    assert '"team":"TX"' in html
    assert "<script src=" not in html          # no external javascript
    assert html.count("<html") == 1


# ---------------- what is and is not measurable ----------------
def test_the_limits_list_shrinks_when_a_map_stream_is_supplied():
    from bgmi_scrim.report import not_measurable
    without = " ".join(not_measurable(False))
    with_map = " ".join(not_measurable(True))
    assert "needs a map stream" in without
    assert "needs a map stream" not in with_map
    assert "flat top-down view" in with_map          # what a map still cannot answer
    assert "Damage dealt or taken" in without and "Damage dealt or taken" in with_map


def test_positional_facts_measure_rotations_edge_play_and_vehicle_time(tmp_path):
    from bgmi_scrim.positional import positional_match
    mev = [
        {"type": "zone_phase", "t": 0.0, "phase": 1, "cx_m": 4000, "cy_m": 4000,
         "r_m": 3000, "closed_t": 60.0, "place": "E5"},
        {"type": "zone_phase", "t": 60.0, "phase": 2, "cx_m": 3500, "cy_m": 3500,
         "r_m": 1500, "closed_t": 150.0, "place": "D4"},
        {"type": "landing", "t": 20.0, "team": "TX", "unit": "team", "x_m": 1000,
         "y_m": 1000, "place": "B2", "confidence": 0.9},
        {"type": "loot", "t": 25.0, "t_end": 70.0, "team": "TX", "unit": "team",
         "duration_s": 45.0, "confidence": 0.9},
        {"type": "rotation", "t": 80.0, "t_end": 140.0, "team": "TX", "unit": "team",
         "duration_s": 60.0, "from": "B2", "to": "D4", "distance_m": 900,
         "mode": "vehicle", "phase": 2, "after_announce_s": 20.0,
         "before_close_s": 10.0, "confidence": 0.9},
        {"type": "rotation", "t": 150.0, "t_end": 180.0, "team": "TX", "unit": "team",
         "duration_s": 30.0, "from": "D4", "to": "D4", "distance_m": 250,
         "mode": "foot", "phase": 2, "after_announce_s": 90.0, "confidence": 0.9},
        {"type": "zone_summary", "t": 200.0, "team": "TX", "unit": "team",
         "time_outside_blue_s": 42.0, "edge_time_fraction": 0.5,
         "tracked_from_s": 0.0, "tracked_to_s": 200.0, "confidence": 0.9},
        {"type": "zone_entry", "t": 0.0, "team": "TX", "unit": "team", "phase": 2,
         "state": "never_entered", "confidence": 0.9},
    ]
    tracks = [{"type": "map_info", "map": "ERANGEL", "world_size_m": 8000, "our_team": "TX"}]
    for i in range(6):
        tracks.append({"type": "sample", "t": i * 20.0, "track": 1, "team": "TX",
                       "x_m": 1000 + i * 400, "y_m": 1000 + i * 400, "confidence": 0.9})
    pm = positional_match(1, _write(tmp_path, "t.jsonl", tracks),
                          _write(tmp_path, "me.jsonl", mev), "TX")
    assert pm.landing["place"] == "B2" and pm.loot_s == 45.0
    assert pm.vehicle_time_s == 60.0 and pm.foot_time_s == 30.0
    assert pm.vehicle_share == pytest.approx(60 / 90)
    assert pm.rotation_distance_m == 1150
    assert pm.edge_fraction == 0.5 and pm.time_outside_blue_s == 42.0
    assert [p["phase"] for p in pm.positions] == [1, 2]
    assert pm.positions[0]["place"] == "B2"
    assert any(e.get("state") == "never_entered" for e in pm.zone_entries)


def test_report_measures_positioning_only_when_the_map_logs_are_there(tmp_path):
    from bgmi_scrim.report import build_report
    ev = _match(tmp_path)
    plain = build_report([ev], team="TX")
    assert "needs a map stream" in plain
    assert "No map stream was supplied" in plain
    mev = [{"type": "zone_phase", "t": 0.0, "phase": 1, "cx_m": 4000, "cy_m": 4000,
            "r_m": 3000, "closed_t": 60.0, "place": "E5"},
           {"type": "landing", "t": 20.0, "team": "TX", "unit": "team", "x_m": 1000,
            "y_m": 1000, "place": "B2", "confidence": 0.9},
           {"type": "rotation", "t": 30.0, "t_end": 50.0, "team": "TX", "unit": "team",
            "duration_s": 20.0, "from": "B2", "to": "C3", "distance_m": 400,
            "mode": "foot", "phase": 1, "after_announce_s": 30.0,
            "before_close_s": 10.0, "confidence": 0.9}]
    rich = build_report([ev], team="TX", map_events_paths=[_write(tmp_path, "m.jsonl", mev)])
    assert "## 5. Positioning and rotations" in rich
    assert "needs a map stream" not in rich
    assert "vehicle time" in rich and "zone-edge time" in rich
    assert "B2 → C3" in rich
