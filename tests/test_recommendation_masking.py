import unittest

import numpy as np

from backend.recommender_service import rank_available_recommendations


class RecommendationMaskingTests(unittest.TestCase):
    def test_preban_high_probability_is_masked_and_probabilities_renormalize(self) -> None:
        scores = np.array([0.0, 0.90, 0.05, 0.05], dtype=np.float64)
        id_to_hero = ["<PAD>", "c_preban", "c_available_a", "c_available_b"]
        hero_to_id = {code: idx for idx, code in enumerate(id_to_hero)}

        heroes, rates = rank_available_recommendations(
            scores,
            id_to_hero=id_to_hero,
            hero_to_id=hero_to_id,
            unavailable_heroes={"c_preban"},
            k=3,
        )

        self.assertEqual(heroes, ["c_available_b", "c_available_a"])
        self.assertNotIn("c_preban", heroes)
        self.assertEqual(rates, [50.0, 50.0])

    def test_candidate_mask_removes_hero_before_rate_normalization(self) -> None:
        scores = np.array([0.0, 0.30, 0.30, 0.40], dtype=np.float64)
        candidate_mask = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
        id_to_hero = ["<PAD>", "c_available_a", "c_masked", "c_available_b"]
        hero_to_id = {code: idx for idx, code in enumerate(id_to_hero)}

        heroes, rates = rank_available_recommendations(
            scores,
            id_to_hero=id_to_hero,
            hero_to_id=hero_to_id,
            unavailable_heroes=set(),
            candidate_mask=candidate_mask,
            k=3,
        )

        self.assertEqual(heroes, ["c_available_b", "c_available_a"])
        self.assertNotIn("c_masked", heroes)
        np.testing.assert_allclose(rates, [40.0 / 70.0 * 100.0, 30.0 / 70.0 * 100.0])


if __name__ == "__main__":
    unittest.main()
