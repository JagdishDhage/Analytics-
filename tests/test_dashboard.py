"""The dashboard's data model: it must carry what the logs say and nothing else."""
from __future__ import annotations

import json

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
