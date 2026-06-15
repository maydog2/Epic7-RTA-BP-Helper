import unittest

import backend.first_pick_recommender as fp


class FirstPickRecommenderTests(unittest.TestCase):
    def setUp(self) -> None:
        fp._first_pick_records = None

    def _set_records(self, records: list[fp.FirstPickRecord]) -> None:
        fp._first_pick_records = records

    def test_derive_directional_prebans_respects_first_pick_side(self) -> None:
        self.assertEqual(fp.derive_directional_prebans("ally", ["A", "B"], ["C", "D"]), (("A", "B"), ("C", "D")))
        self.assertEqual(fp.derive_directional_prebans("enemy", ["A", "B"], ["C", "D"]), (("C", "D"), ("A", "B")))

    def test_directional_contexts_are_not_identical_when_swapped(self) -> None:
        context_a = fp.derive_directional_prebans("ally", ["A", "B"], ["C", "D"])
        context_b = fp.derive_directional_prebans("ally", ["C", "D"], ["A", "B"])
        self.assertNotEqual(context_a, context_b)

    def test_level_one_exact_directional_match(self) -> None:
        self._set_records(
            [
                fp.FirstPickRecord("ally", ("A", "B"), ("C", "D"), "c_target"),
                fp.FirstPickRecord("ally", ("A", "B"), ("C", "D"), "c_noise"),
                fp.FirstPickRecord("ally", ("C", "D"), ("A", "B"), "c_wrong_context"),
            ]
        )

        result = fp.recommend_first_pick(
            ally_preban=["A", "B"],
            enemy_preban=["C", "D"],
            first_pick_side="ally",
            excluded_heroes={"A", "B", "C", "D"},
            top_k=3,
        )

        self.assertEqual(result["first_pick_fallback_level"], 1)
        self.assertEqual(result["top_10_heroes"][:2], ["c_target", "c_noise"])
        for hero in ("A", "B", "C", "D"):
            self.assertNotIn(hero, result["top_10_heroes"])

    def test_level_two_exact_first_side_and_partial_second_side(self) -> None:
        self._set_records(
            [
                fp.FirstPickRecord("ally", ("A", "B"), ("C", "X"), "c_level2"),
                fp.FirstPickRecord("ally", ("A", "B"), ("D", "Y"), "c_level2b"),
                fp.FirstPickRecord("ally", ("X", "Y"), ("C", "D"), "c_not_level2"),
            ]
        )

        result = fp.recommend_first_pick(
            ally_preban=["A", "B"],
            enemy_preban=["C", "D"],
            first_pick_side="ally",
            excluded_heroes={"A", "B", "C", "D"},
            top_k=5,
        )

        self.assertEqual(result["first_pick_fallback_level"], 2)
        self.assertEqual(set(result["top_10_heroes"][:2]), {"c_level2", "c_level2b"})
        self.assertAlmostEqual(sum(result["top_10_rates"]), 100.0)

    def test_sparse_exact_match_backfills_from_lower_levels(self) -> None:
        self._set_records(
            [
                fp.FirstPickRecord("ally", ("A", "B"), ("C", "D"), "c_exact"),
                fp.FirstPickRecord("ally", ("A", "B"), ("C", "X"), "c_level2"),
                fp.FirstPickRecord("ally", ("A", "X"), ("C", "Y"), "c_level3"),
                fp.FirstPickRecord("ally", ("P", "Q"), ("R", "S"), "c_global"),
            ]
        )

        result = fp.recommend_first_pick(
            ally_preban=["A", "B"],
            enemy_preban=["C", "D"],
            first_pick_side="ally",
            excluded_heroes={"A", "B", "C", "D"},
            top_k=4,
        )

        self.assertEqual(result["first_pick_fallback_level"], 1)
        self.assertEqual(result["first_pick_filled_through_level"], 4)
        self.assertEqual(set(result["top_10_heroes"]), {"c_exact", "c_level2", "c_level3", "c_global"})
        self.assertEqual(result["top_10_rates"], sorted(result["top_10_rates"], reverse=True))

    def test_level_four_season_global_when_no_directional_similarity(self) -> None:
        self._set_records(
            [
                fp.FirstPickRecord("enemy", ("X", "Y"), ("Z", "W"), "c_global_a"),
                fp.FirstPickRecord("enemy", ("P", "Q"), ("R", "S"), "c_global_a"),
                fp.FirstPickRecord("enemy", ("P", "Q"), ("R", "S"), "c_global_b"),
            ]
        )

        result = fp.recommend_first_pick(
            ally_preban=["A", "B"],
            enemy_preban=["C", "D"],
            first_pick_side="enemy",
            excluded_heroes={"A", "B", "C", "D"},
            top_k=3,
        )

        self.assertEqual(result["first_pick_fallback_level"], 4)
        self.assertEqual(result["top_10_heroes"][0], "c_global_a")
        self.assertEqual(result["top_10_rates"], sorted(result["top_10_rates"], reverse=True))

    def test_preban_hero_with_high_probability_is_excluded_from_percentages(self) -> None:
        self._set_records(
            [
                fp.FirstPickRecord("ally", ("A", "B"), ("C", "D"), "A"),
                fp.FirstPickRecord("ally", ("A", "B"), ("C", "D"), "c_pick_a"),
                fp.FirstPickRecord("ally", ("A", "B"), ("C", "D"), "c_pick_b"),
            ]
        )

        result = fp.recommend_first_pick(
            ally_preban=["A", "B"],
            enemy_preban=["C", "D"],
            first_pick_side="ally",
            excluded_heroes={"A", "B", "C", "D"},
            top_k=5,
        )

        self.assertNotIn("A", result["top_10_heroes"])
        self.assertEqual(result["top_10_heroes"], ["c_pick_a", "c_pick_b"])
        self.assertEqual(result["top_10_rates"], [50.0, 50.0])


if __name__ == "__main__":
    unittest.main()
