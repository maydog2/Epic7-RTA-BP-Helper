import unittest

from match_history_utils import enrich_match_draft, get_position_bucket, make_draft_entry


class PositionBucketTests(unittest.TestCase):
    def test_get_position_bucket_mapping(self) -> None:
        self.assertEqual(get_position_bucket(1), "1")
        self.assertEqual(get_position_bucket(2), "2_3")
        self.assertEqual(get_position_bucket(3), "2_3")
        self.assertEqual(get_position_bucket(4), "4")
        self.assertEqual(get_position_bucket(5), "5_protected")
        self.assertEqual(get_position_bucket(6), "6_protected")
        self.assertEqual(get_position_bucket(7), "7")
        self.assertEqual(get_position_bucket(8), "8_9")
        self.assertEqual(get_position_bucket(9), "8_9")
        self.assertEqual(get_position_bucket(10), "10")

    def test_make_draft_entry_and_enrich_match_draft(self) -> None:
        entry = make_draft_entry(order=5, side="ally", hero="c1001")
        self.assertEqual(entry["position_bucket"], "5_protected")

        match = enrich_match_draft(
            {
                "match_id": 1,
                "first_pick_side": "ally",
                "winner_side": "enemy",
                "ally_preban": [],
                "enemy_preban": [],
                "draft": [{"order": 7, "side": "enemy", "hero": "c2002"}],
            }
        )
        self.assertEqual(match["draft"][0]["position_bucket"], "7")


if __name__ == "__main__":
    unittest.main()
