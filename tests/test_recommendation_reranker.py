import unittest
from unittest.mock import patch

from backend.recommendation_reranker import (
    HeroAppearanceStat,
    apply_low_pick_top3_guard,
    evaluate_low_pick_support,
    reorder_with_low_pick_top3_guard,
    rerank_candidates,
    reset_cached_stats,
)


class LowPickSupportTests(unittest.TestCase):
    def test_model_confidence_allows_top3(self) -> None:
        evidence = evaluate_low_pick_support(
            model_score_norm=0.40,
            ally_synergy_score=0.0,
            enemy_response_score=0.0,
        )
        self.assertTrue(evidence["model_confident"])
        self.assertTrue(any(evidence.values()))

    def test_response_support_allows_top3(self) -> None:
        evidence = evaluate_low_pick_support(
            model_score_norm=0.10,
            ally_synergy_score=0.0,
            enemy_response_score=0.35,
        )
        self.assertTrue(evidence["response_supported"])

    def test_synergy_support_allows_top3(self) -> None:
        evidence = evaluate_low_pick_support(
            model_score_norm=0.10,
            ally_synergy_score=0.31,
            enemy_response_score=0.0,
        )
        self.assertTrue(evidence["synergy_supported"])


class LowPickTop3GuardOrderingTests(unittest.TestCase):
    def test_unsupported_low_pick_moves_below_top3(self) -> None:
        ordered = ["low", "a", "b", "c", "d"]
        top3_blocked = {
            "low": True,
            "a": False,
            "b": False,
            "c": False,
            "d": False,
        }
        result = reorder_with_low_pick_top3_guard(ordered, top3_blocked)
        self.assertEqual(result[:3], ["a", "b", "c"])
        self.assertIn("low", result[3:])
        self.assertEqual(len(result), 5)

    def test_blocked_fills_top3_when_fewer_than_three_eligible(self) -> None:
        ordered = ["blocked1", "blocked2", "eligible", "blocked3"]
        top3_blocked = {
            "blocked1": True,
            "blocked2": True,
            "eligible": False,
            "blocked3": True,
        }
        result = reorder_with_low_pick_top3_guard(ordered, top3_blocked)
        self.assertEqual(result[:3], ["eligible", "blocked1", "blocked2"])
        self.assertEqual(result[3:], ["blocked3"])

    def test_guard_does_not_introduce_new_candidates(self) -> None:
        ordered = ["a", "b", "c"]
        top3_blocked = {"a": False, "b": True, "c": False}
        result = reorder_with_low_pick_top3_guard(ordered, top3_blocked)
        self.assertEqual(set(result), set(ordered))
        self.assertEqual(len(result), len(ordered))

    def test_apply_guard_adds_debug_fields(self) -> None:
        ordered = ["low", "safe"]
        component_scores = {
            "low": {
                "model_score": 0.10,
                "ally_synergy_score": 0.0,
                "enemy_response_score": 0.0,
                "final_score": 0.10,
            },
            "safe": {
                "model_score": 0.20,
                "ally_synergy_score": 0.0,
                "enemy_response_score": 0.0,
                "final_score": 0.20,
            },
        }
        debug_records = [
            {
                "hero": "low",
                "model_score": 0.10,
                "ally_synergy_score": 0.0,
                "enemy_response_score": 0.0,
                "final_score": 0.10,
                "position_bucket": "4",
                "reason": {},
            },
            {
                "hero": "safe",
                "model_score": 0.20,
                "ally_synergy_score": 0.0,
                "enemy_response_score": 0.0,
                "final_score": 0.20,
                "position_bucket": "4",
                "reason": {},
            },
        ]
        appearance_stats = {
            "low": HeroAppearanceStat(appearance_count=10, appearance_rate=0.01),
            "safe": HeroAppearanceStat(appearance_count=200, appearance_rate=0.20),
        }

        with patch("backend.recommendation_reranker.low_pick_guard_enabled", return_value=True):
            with patch(
                "backend.recommendation_reranker.load_hero_appearance_stats",
                return_value=appearance_stats,
            ):
                final_ordered, final_debug = apply_low_pick_top3_guard(
                    ordered=ordered,
                    component_scores=component_scores,
                    debug_records=debug_records,
                )

        self.assertEqual(final_ordered[0], "safe")
        self.assertIn("low", final_ordered)
        low_debug = next(record for record in final_debug if record["hero"] == "low")
        self.assertTrue(low_debug["is_low_pick"])
        self.assertTrue(low_debug["top3_blocked"])
        self.assertEqual(low_debug["appearance_count"], 10)


class RerankCandidatesGuardIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_cached_stats()

    def tearDown(self) -> None:
        reset_cached_stats()

    def _rerank_with_appearance_stats(
        self,
        *,
        candidates: list[str],
        model_scores: dict[str, float],
        appearance_stats: dict[str, HeroAppearanceStat],
        synergy_lookup: dict[tuple[str, str], object] | None = None,
        response_lookup: dict[tuple[str, str, str], object] | None = None,
    ) -> dict[str, object]:
        synergy_lookup = synergy_lookup or {}
        response_lookup = response_lookup or {}

        with patch("backend.recommendation_reranker.low_pick_guard_enabled", return_value=True):
            with patch(
                "backend.recommendation_reranker.load_hero_appearance_stats",
                return_value=appearance_stats,
            ):
                with patch("backend.recommendation_reranker.load_synergy_lookup", return_value=synergy_lookup):
                    with patch("backend.recommendation_reranker.load_response_lookup", return_value=response_lookup):
                        with patch("backend.recommendation_reranker.load_opening_response_lookup", return_value={}):
                            return rerank_candidates(
                                candidates=candidates,
                                model_scores=model_scores,
                                ally_picks=["ally1"],
                                enemy_picks=["enemy1"],
                                unavailable_heroes=set(),
                                position_bucket="4",
                                first_pick_team="My Team",
                                top_k=10,
                            )

    def test_low_pick_unsupported_not_removed_from_top10(self) -> None:
        candidates = ["low", "a", "b", "c", "d", "e", "f", "g", "h", "i"]
        model_scores = {hero: 1.0 - index * 0.05 for index, hero in enumerate(candidates)}
        appearance_stats = {
            hero: HeroAppearanceStat(appearance_count=200, appearance_rate=0.1) for hero in candidates
        }
        appearance_stats["low"] = HeroAppearanceStat(appearance_count=5, appearance_rate=0.001)

        result = self._rerank_with_appearance_stats(
            candidates=candidates,
            model_scores=model_scores,
            appearance_stats=appearance_stats,
        )

        self.assertEqual(len(result["top_10_heroes"]), 10)
        self.assertIn("low", result["top_10_heroes"])
        self.assertNotIn("low", result["top_10_heroes"][:3])

    def test_low_pick_with_high_response_can_stay_in_top3(self) -> None:
        from backend.recommendation_reranker import ResponseEntry

        candidates = ["low", "a", "b", "c"]
        model_scores = {"low": 0.9, "a": 0.5, "b": 0.4, "c": 0.3}
        appearance_stats = {
            "low": HeroAppearanceStat(appearance_count=5, appearance_rate=0.001),
            "a": HeroAppearanceStat(appearance_count=200, appearance_rate=0.2),
            "b": HeroAppearanceStat(appearance_count=200, appearance_rate=0.2),
            "c": HeroAppearanceStat(appearance_count=200, appearance_rate=0.2),
        }
        response_lookup = {
            ("4", "low", "enemy1"): ResponseEntry(score=0.40, response_count=10),
        }

        result = self._rerank_with_appearance_stats(
            candidates=candidates,
            model_scores=model_scores,
            appearance_stats=appearance_stats,
            response_lookup=response_lookup,
        )

        self.assertIn("low", result["top_10_heroes"][:3])

    def test_low_pick_with_high_synergy_can_stay_in_top3(self) -> None:
        from backend.recommendation_reranker import SynergyEntry

        candidates = ["low", "a", "b", "c"]
        model_scores = {"low": 0.9, "a": 0.5, "b": 0.4, "c": 0.3}
        appearance_stats = {
            "low": HeroAppearanceStat(appearance_count=5, appearance_rate=0.001),
            "a": HeroAppearanceStat(appearance_count=200, appearance_rate=0.2),
            "b": HeroAppearanceStat(appearance_count=200, appearance_rate=0.2),
            "c": HeroAppearanceStat(appearance_count=200, appearance_rate=0.2),
        }
        synergy_lookup = {
            ("low", "ally1"): SynergyEntry(score=0.35, same_team_count=10),
        }

        result = self._rerank_with_appearance_stats(
            candidates=candidates,
            model_scores=model_scores,
            appearance_stats=appearance_stats,
            synergy_lookup=synergy_lookup,
        )

        self.assertIn("low", result["top_10_heroes"][:3])

    def test_low_pick_with_high_model_score_can_stay_in_top3(self) -> None:
        candidates = ["low", "a", "b", "c"]
        model_scores = {"low": 0.90, "a": 0.10, "b": 0.09, "c": 0.08}
        appearance_stats = {
            "low": HeroAppearanceStat(appearance_count=5, appearance_rate=0.001),
            "a": HeroAppearanceStat(appearance_count=200, appearance_rate=0.2),
            "b": HeroAppearanceStat(appearance_count=200, appearance_rate=0.2),
            "c": HeroAppearanceStat(appearance_count=200, appearance_rate=0.2),
        }

        result = self._rerank_with_appearance_stats(
            candidates=candidates,
            model_scores=model_scores,
            appearance_stats=appearance_stats,
        )

        self.assertEqual(result["top_10_heroes"][0], "low")

    def test_guard_can_be_disabled(self) -> None:
        candidates = ["low", "a", "b", "c"]
        model_scores = {"low": 0.9, "a": 0.5, "b": 0.4, "c": 0.3}
        appearance_stats = {
            "low": HeroAppearanceStat(appearance_count=5, appearance_rate=0.001),
            "a": HeroAppearanceStat(appearance_count=200, appearance_rate=0.2),
            "b": HeroAppearanceStat(appearance_count=200, appearance_rate=0.2),
            "c": HeroAppearanceStat(appearance_count=200, appearance_rate=0.2),
        }

        with patch("backend.recommendation_reranker.low_pick_guard_enabled", return_value=False):
            with patch(
                "backend.recommendation_reranker.load_hero_appearance_stats",
                return_value=appearance_stats,
            ):
                with patch("backend.recommendation_reranker.load_synergy_lookup", return_value={}):
                    with patch("backend.recommendation_reranker.load_response_lookup", return_value={}):
                        with patch("backend.recommendation_reranker.load_opening_response_lookup", return_value={}):
                            result = rerank_candidates(
                                candidates=candidates,
                                model_scores=model_scores,
                                ally_picks=["ally1"],
                                enemy_picks=["enemy1"],
                                unavailable_heroes=set(),
                                position_bucket="4",
                                first_pick_team="My Team",
                                top_k=10,
                            )

        self.assertEqual(result["top_10_heroes"][0], "low")


if __name__ == "__main__":
    unittest.main()
