import json
import sys
import threading
import unittest
from pathlib import Path

WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "workflow_scripts"
if str(WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_DIR))

import get_matches as gm  # noqa: E402
from match_history_utils import (  # noqa: E402
    ALLY_FINAL_BAN_TARGET,
    ENEMY_FINAL_BAN_TARGET,
    make_draft_entry,
)


class ScraperCoordinatorTests(unittest.TestCase):
    def test_load_state_prefers_visited_players(self) -> None:
        path = Path("data/test_scraper_state_players.json")
        path.write_text(
            json.dumps({"visited_players": ["A ( Global )"], "visited_heroes": ["B ( Korea )"]}),
            encoding="utf-8",
        )
        old_path = gm.SCRAPER_STATE_PATH
        try:
            gm.SCRAPER_STATE_PATH = path
            coord = gm.ScraperCoordinator()
            coord.load_scraper_state()
            self.assertIn("A ( Global )", coord.visited_players)
            self.assertNotIn("B ( Korea )", coord.visited_players)
            self.assertEqual(coord.in_progress_players, set())
        finally:
            gm.SCRAPER_STATE_PATH = old_path
            path.unlink(missing_ok=True)

    def test_load_state_fallback_visited_heroes(self) -> None:
        path = Path("data/test_scraper_state_heroes.json")
        path.write_text(json.dumps({"visited_heroes": ["Legacy ( Asia )"]}), encoding="utf-8")
        old_path = gm.SCRAPER_STATE_PATH
        try:
            gm.SCRAPER_STATE_PATH = path
            coord = gm.ScraperCoordinator()
            coord.load_scraper_state()
            self.assertIn("Legacy ( Asia )", coord.visited_players)
        finally:
            gm.SCRAPER_STATE_PATH = old_path
            path.unlink(missing_ok=True)

    def test_match_id_allocated_under_lock(self) -> None:
        coord = gm.ScraperCoordinator()
        ids: list[int] = []
        lock = threading.Lock()

        def worker(worker_idx: int) -> None:
            draft = [
                make_draft_entry(
                    order=i,
                    side="ally" if i <= 5 else "enemy",
                    hero=f"c{worker_idx:04d}{i:02d}",
                )
                for i in range(1, 11)
            ]
            match = {
                "first_pick_side": "ally",
                "winner_side": "ally",
                "ally_preban": [],
                "enemy_preban": [],
                "draft": draft,
            }
            result = coord.record_match(match)
            if result == "saved":
                with lock:
                    ids.append(match["match_id"])

        threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(ids), 8)
        self.assertEqual(len(set(ids)), 8)
        self.assertEqual(coord.next_match_id, 8)

    def test_claim_skips_in_progress_and_visited(self) -> None:
        coord = gm.ScraperCoordinator()
        coord.visited_players.add("Done ( Global )")
        coord.in_progress_players.add("Busy ( Korea )")

        self.assertFalse(coord.try_claim_player("Done ( Global )"))
        self.assertFalse(coord.try_claim_player("Busy ( Korea )"))
        self.assertTrue(coord.try_claim_player("Free ( Europe )"))
        self.assertIn("Free ( Europe )", coord.in_progress_players)

        coord.complete_player_failure("Free ( Europe )")
        self.assertNotIn("Free ( Europe )", coord.in_progress_players)
        self.assertNotIn("Free ( Europe )", coord.visited_players)

        coord.try_claim_player("Free ( Europe )")
        coord.complete_player_success("Free ( Europe )")
        self.assertIn("Free ( Europe )", coord.visited_players)
        self.assertNotIn("Free ( Europe )", coord.in_progress_players)

    def test_record_match_updates_existing_row_with_final_ban_targets(self) -> None:
        coord = gm.ScraperCoordinator()
        draft = [
            make_draft_entry(order=1, side="ally", hero="c1001"),
            make_draft_entry(order=2, side="enemy", hero="c2001"),
            make_draft_entry(order=3, side="enemy", hero="c2002"),
            make_draft_entry(order=4, side="ally", hero="c1002"),
            make_draft_entry(order=5, side="ally", hero="c1003"),
            make_draft_entry(order=6, side="enemy", hero="c2003"),
            make_draft_entry(order=7, side="enemy", hero="c2004"),
            make_draft_entry(order=8, side="ally", hero="c1004"),
            make_draft_entry(order=9, side="ally", hero="c1005"),
            make_draft_entry(order=10, side="enemy", hero="c2005"),
        ]
        existing = {
            "match_id": 0,
            "first_pick_side": "ally",
            "winner_side": "ally",
            "ally_preban": ["c3001", "c3002"],
            "enemy_preban": ["c4001", "c4002"],
            "draft": draft,
        }
        incoming = {
            "first_pick_side": "ally",
            "winner_side": "ally",
            "ally_preban": ["c3001", "c3002"],
            "enemy_preban": ["c4001", "c4002"],
            ALLY_FINAL_BAN_TARGET: "c2004",
            ENEMY_FINAL_BAN_TARGET: "c1004",
            "draft": draft,
        }

        self.assertEqual(coord.record_match(existing), "saved")
        self.assertEqual(len(coord.matches), 1)
        self.assertNotIn(ALLY_FINAL_BAN_TARGET, coord.matches[0])

        self.assertEqual(coord.record_match(incoming), "updated")
        self.assertEqual(len(coord.matches), 1)
        self.assertEqual(coord.matches[0][ALLY_FINAL_BAN_TARGET], "c2004")
        self.assertEqual(coord.matches[0][ENEMY_FINAL_BAN_TARGET], "c1004")
        self.assertEqual(coord.matches[0]["match_id"], 0)


if __name__ == "__main__":
    unittest.main()
