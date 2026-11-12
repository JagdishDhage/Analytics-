"""The scenario input: a feed that names players, with real tags ending in L."""
import sys; sys.path.insert(0, "/home/user/Analytics-")
from bgmi_scrim.profile import Profile
from bgmi_scrim.killfeed import _actor, parse_feed_line

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
