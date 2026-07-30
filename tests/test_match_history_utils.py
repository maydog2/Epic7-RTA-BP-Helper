import unittest
from datetime import datetime, timedelta
from pathlib import Path

from bs4 import BeautifulSoup

import sys

WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "workflow_scripts"
if str(WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_DIR))

from match_history_utils import (  # noqa: E402
    ALLY_FINAL_BAN_TARGET,
    ENEMY_FINAL_BAN_TARGET,
    extract_played_at,
    extract_preban_and_picks,
    extract_season_patch,
    extract_team_draft,
    extract_warfare_rules,
    is_match_within_days,
    make_draft_entry,
    parse_relative_time_text,
    validate_match_record,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "battle_snippet.html"


def _baseline_match(**overrides) -> dict:
    draft = [
        make_draft_entry(order=1, side="ally", hero="c1003"),
        make_draft_entry(order=2, side="enemy", hero="c2003"),
        make_draft_entry(order=3, side="enemy", hero="c2004"),
        make_draft_entry(order=4, side="ally", hero="c1004"),
        make_draft_entry(order=5, side="ally", hero="c1005"),
        make_draft_entry(order=6, side="enemy", hero="c2005"),
        make_draft_entry(order=7, side="enemy", hero="c2006"),
        make_draft_entry(order=8, side="ally", hero="c1006"),
        make_draft_entry(order=9, side="ally", hero="c1007"),
        make_draft_entry(order=10, side="enemy", hero="c2007"),
    ]
    match = {
        "match_id": 1,
        "first_pick_side": "ally",
        "winner_side": "ally",
        "ally_preban": ["c1001", "c1002"],
        "enemy_preban": ["c2001", "c2002"],
        "draft": draft,
    }
    match.update(overrides)
    return match


class MatchHistoryUtilsTests(unittest.TestCase):
    def test_parse_relative_time_text(self) -> None:
        now = datetime(2026, 5, 20, 12, 0, 0)

        self.assertEqual(parse_relative_time_text("a day ago", now=now), now - timedelta(days=1))
        self.assertEqual(parse_relative_time_text("2 days ago", now=now), now - timedelta(days=2))
        self.assertEqual(parse_relative_time_text("an hour ago", now=now), now - timedelta(hours=1))
        self.assertEqual(parse_relative_time_text("just now", now=now), now)
        self.assertIsNone(parse_relative_time_text("Time 04:11", now=now))

    def test_extract_battle_fields_from_fixture(self) -> None:
        now = datetime(2026, 5, 20, 12, 0, 0)
        battle = BeautifulSoup(FIXTURE_PATH.read_text(encoding="utf-8"), "html.parser").select_one(
            "li.battle-info"
        )
        self.assertIsNotNone(battle)

        played_at, played_at_text = extract_played_at(battle, now=now)
        self.assertEqual(played_at_text, "a day ago")
        self.assertEqual(played_at, now - timedelta(days=1))
        self.assertTrue(is_match_within_days(played_at, 2, now=now))
        self.assertFalse(is_match_within_days(played_at, 0, now=now))
        self.assertEqual(extract_warfare_rules(battle), {"warfare_rules": "Resistance"})

        my_team = battle.find("div", class_="my-team w-100")
        enemy_team = battle.find("div", class_="enemy-team w-100")
        ally_preban, ally_picks = extract_preban_and_picks(my_team)
        enemy_preban, enemy_picks = extract_preban_and_picks(enemy_team)
        ally_extraction = extract_team_draft(my_team)
        enemy_extraction = extract_team_draft(enemy_team)

        self.assertEqual(len(ally_preban), 2)
        self.assertEqual(len(enemy_preban), 2)
        self.assertEqual(len(ally_picks), 5)
        self.assertEqual(len(enemy_picks), 5)
        self.assertEqual(ally_extraction.final_banned_pick, "c1006")
        self.assertEqual(enemy_extraction.final_banned_pick, "c2007")
        self.assertIsNone(ally_extraction.error)
        self.assertIsNone(enemy_extraction.error)
        self.assertEqual(extract_season_patch(battle), {"season": "Season 1", "patch": "v1.0"})

    def test_validate_match_without_final_ban_fields(self) -> None:
        self.assertEqual(validate_match_record(_baseline_match()), [])

    def test_validate_match_with_valid_final_ban_fields(self) -> None:
        match = _baseline_match(
            **{
                ALLY_FINAL_BAN_TARGET: "c2007",
                ENEMY_FINAL_BAN_TARGET: "c1006",
            }
        )
        self.assertEqual(validate_match_record(match), [])

    def test_validate_match_rejects_partial_final_ban_fields(self) -> None:
        match = _baseline_match(**{ALLY_FINAL_BAN_TARGET: "c2007"})
        errors = validate_match_record(match)
        self.assertTrue(any("must both be present or absent" in error for error in errors))

    def test_validate_match_rejects_wrong_side_final_ban_target(self) -> None:
        match = _baseline_match(
            **{
                ALLY_FINAL_BAN_TARGET: "c1006",
                ENEMY_FINAL_BAN_TARGET: "c2007",
            }
        )
        errors = validate_match_record(match)
        self.assertTrue(any("not in enemy draft picks" in error for error in errors))
        self.assertTrue(any("not in ally draft picks" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
