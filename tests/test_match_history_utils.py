import unittest
from datetime import datetime, timedelta
from pathlib import Path

from bs4 import BeautifulSoup

from match_history_utils import (
    extract_played_at,
    extract_preban_and_picks,
    extract_season_patch,
    extract_warfare_rules,
    is_match_within_days,
    parse_relative_time_text,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "battle_snippet.html"


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

        self.assertEqual(len(ally_preban), 2)
        self.assertEqual(len(enemy_preban), 2)
        self.assertEqual(len(ally_picks), 5)
        self.assertEqual(len(enemy_picks), 5)
        self.assertEqual(extract_season_patch(battle), {"season": "Season 1", "patch": "v1.0"})


if __name__ == "__main__":
    unittest.main()
