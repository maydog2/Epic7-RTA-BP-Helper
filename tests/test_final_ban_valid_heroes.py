import unittest

from backend.final_ban_recommender import build_ban_candidates, derive_ordered_picks
from backend.recommender_service import load_final_ban_valid_heroes


class FinalBanValidHeroesTests(unittest.TestCase):
    def test_c5069_in_hero_details_valid_set(self) -> None:
        valid_heroes = load_final_ban_valid_heroes()
        self.assertIn("c5069", valid_heroes)

    def test_new_hero_not_filtered_from_ban_candidates(self) -> None:
        valid_heroes = load_final_ban_valid_heroes()
        ally = derive_ordered_picks(
            ["c1001", "c1002", "c1003", "c1004", "c1005"],
            first_pick_team="My Team",
            side="ally",
        )
        enemy = derive_ordered_picks(
            ["c2001", "c2002", "c2003", "c2004", "c5069"],
            first_pick_team="My Team",
            side="enemy",
        )

        candidates, filtered_out = build_ban_candidates(
            ally_picks=ally,
            enemy_picks=enemy,
            first_pick_team="My Team",
            valid_heroes=valid_heroes,
        )

        candidate_heroes = {str(pick["hero"]) for pick in candidates}
        self.assertIn("c5069", candidate_heroes)
        self.assertFalse(
            any(item["hero"] == "c5069" and item["reason"] == "invalid_hero" for item in filtered_out),
        )


if __name__ == "__main__":
    unittest.main()
