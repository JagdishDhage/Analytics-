"""The scenario input: a feed that names players, with real tags ending in L."""
import sys; sys.path.insert(0, "/home/user/Analytics-")
from bgmi_scrim.profile import Profile
from bgmi_scrim.killfeed import _actor, _actor_by_name, parse_feed_line

PROF = Profile.from_dict({
    "name": "t", "resolution": [1920, 1080],
    "team_tags": ["SOUL", "GODL", "TX", "ENT", "K9"],
    "roster": {"SOUL": ["VIPER", "OWAIS", "HECTOR", "AKOP"],
               "GODL": ["JONATHAN", "CLUTCHGOD", "NOVO", "SPRAYGOD"]},
    "rois": {},
})

def test_trailing_letter_is_not_a_slot():
    # 'L' closes the tag SOUL; it must not become slot 1
    assert _actor("SOULXVIPER", PROF.team_tags, "first")[:2] == ("SOUL", None)
    assert _actor("GODLXJONATHAN", PROF.team_tags, "first")[:2] == ("GODL", None)

def test_explicit_slot_still_read():
    assert _actor("SOUL#1", PROF.team_tags, "first")[:2] == ("SOUL", 1)
    assert _actor("GODL 4", PROF.team_tags, "first")[:2] == ("GODL", 4)
    assert _actor("TX#3", PROF.team_tags, "first")[:2] == ("TX", 3)

def test_roster_supplies_slot_for_named_feed():
    e = parse_feed_line("SOULXVIPER FINISHED GODLXJONATHAN", 0.9, PROF)
    assert (e.kind, e.victim_team, e.victim_slot) == ("kill", "GODL", 1)
    assert (e.killer_team, e.killer_slot) == ("SOUL", 1)

def test_unknown_player_gives_no_slot_rather_than_a_guess():
    e = parse_feed_line("SOULXSUBSTITUTE KNOCKED OUT GODLXTRIALIST", 0.9, PROF)
    assert e.victim_team == "GODL" and e.victim_slot is None

def test_ocr_lost_spaces_still_parse():
    e = parse_feed_line("SOULXVIPERFINISHEDGODLXNOVO", 0.5, PROF)
    assert (e.kind, e.victim_team, e.victim_slot) == ("kill", "GODL", 3)


SHORT_VS_LONG = Profile.from_dict({
    "name": "t2", "resolution": [1920, 1080],
    "team_tags": ["OR", "ORCA", "BLIND", "TX", "IQOO", "GB"],
    "roster": {t: ["ALFA", "BRAVO", "CHARL", "DELTA"]
               for t in ["OR", "ORCA", "BLIND", "TX", "IQOO", "GB"]},
    "rois": {},
})


def test_a_short_tag_does_not_absorb_a_longer_one():
    # a sliding window finds "DxBRAVO" inside BLINDxBRAVO and scores ORxBRAVO
    # at 0.80; whole-token matching scores it 0.63 and keeps BLIND
    for label, team in (("BLINDXBRAVO", "BLIND"), ("ORXBRAVO", "OR"),
                        ("ORCAXBRAVO", "ORCA"), ("IQOOXDELTA", "IQOO"),
                        ("TXXDELTA", "TX")):
        got = _actor_by_name(label, SHORT_VS_LONG, "first")
        assert got[0] == team, f"{label} read as {got[0]}"


def test_killer_and_victim_both_keep_their_own_tag():
    e = parse_feed_line("BLINDXBRAVO KNOCKED OUT GBXBRAVO", 0.85, SHORT_VS_LONG)
    assert (e.killer_team, e.killer_slot) == ("BLIND", 2)
    assert (e.victim_team, e.victim_slot) == ("GB", 2)


def test_spaceless_fallback_still_resolves():
    e = parse_feed_line("ORCAXBRAVOFINISHEDBLINDXBRAVO", 0.85, SHORT_VS_LONG)
    assert (e.killer_team, e.victim_team) == ("ORCA", "BLIND")


def test_an_unslotted_read_is_not_a_second_kill():
    """One kill read twice -- once with the slot, once without -- is one
    event, not two. This is the duplicate the gate forbids."""
    from bgmi_scrim.killfeed import FeedTracker
    tr = FeedTracker(prof=PROF, confirm_sightings=2)
    out = []
    # the roster cannot place "SUBSTITUTE", so this read carries no slot
    out += tr.observe(0.0, [("SOULXVIPER FINISHED GODLXSUBSTITUTE", 0.8)])
    out += tr.observe(0.5, [("SOULXVIPER FINISHED GODLXSUBSTITUTE", 0.8)])
    out += tr.observe(1.0, [("SOULXVIPER FINISHED GODLXJONATHAN", 0.9)])
    out += tr.observe(1.5, [("SOULXVIPER FINISHED GODLXJONATHAN", 0.9)])
    kills = [e for e in out if e.kind == "kill" and e.victim_team == "GODL"]
    assert len(kills) == 1, [(e.victim_slot, e.t) for e in kills]


def test_two_slotted_victims_stay_two_events():
    from bgmi_scrim.killfeed import FeedTracker
    tr = FeedTracker(prof=PROF, confirm_sightings=2)
    out = []
    for t in (0.0, 0.5):
        out += tr.observe(t, [("SOULXVIPER FINISHED GODLXJONATHAN", 0.9),
                              ("SOULXVIPER FINISHED GODLXNOVO", 0.9)])
    kills = sorted(e.victim_slot for e in out if e.kind == "kill")
    assert kills == [1, 3], kills


def test_an_unreadable_text_roi_falls_back_to_the_constrained_engine():
    """Our own team's panel row is drawn highlighted. Read with the free
    engine its glyphs fall outside the alphabet, the filter leaves an empty
    string, and the row goes unnamed -- which cost TX its alive count on
    every frame. An empty read retries with the engine constrained."""
    import numpy as np
    import cv2
    from bgmi_scrim.ocr import read_text
    img = np.full((40, 80), 255, np.uint8)
    cv2.putText(img, "TX", (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA)
    txt, _ = read_text(img)
    assert txt, "a legible ROI must not read as nothing"
    assert txt.startswith("T")
