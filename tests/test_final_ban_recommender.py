import unittest
from unittest import mock

import backend.final_ban_recommender as fbr
from match_history_utils import get_position_bucket


def _pick(hero: str, order: int) -> dict[str, object]:
    return {
        "hero": hero,
        "order": order,
        "position_bucket": get_position_bucket(order),
    }


def _set_lookups(
    *,
    synergy: dict[tuple[str, str], fbr.SynergyEntry] | None = None,
    response: dict[tuple[str, str, str], fbr.ResponseEntry] | None = None,
    response_evidence: set[str] | None = None,
) -> None:
    fbr.reset_cached_stats()
    fbr._synergy_lookup = synergy or {}
    fbr._response_lookup = response or {}
    fbr._candidates_with_response_evidence = (
        response_evidence if response_evidence is not None else {enemy for _, _, enemy in (response or {})}
    )


def _sample_draft() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ally = [_pick("c1001", 1), _pick("c1002", 4), _pick("c1003", 5), _pick("c1004", 8), _pick("c1005", 9)]
    enemy = [_pick("c2001", 2), _pick("c2002", 3), _pick("c2003", 6), _pick("c2004", 7), _pick("c2005", 10)]
    return ally, enemy


class FinalBanRecommenderTests(unittest.TestCase):
    def setUp(self) -> None:
        _set_lookups()

    def test_candidates_only_from_enemy_picks_excluding_protected_orders(self) -> None:
        ally, enemy = _sample_draft()
        candidates, filtered_out = fbr.build_ban_candidates(
            ally_picks=ally,
            enemy_picks=enemy,
            first_pick_team="My Team",
            valid_heroes={str(pick["hero"]) for pick in ally + enemy},
        )

        candidate_heroes = {str(pick["hero"]) for pick in candidates}
        self.assertEqual(candidate_heroes, {"c2001", "c2002", "c2004", "c2005"})
        self.assertNotIn("c2003", candidate_heroes)
        self.assertTrue(
            all(item["reason"] == "protected_order" for item in filtered_out if item["hero"] == "c2003")
        )

    def test_never_recommends_ally_or_prebanned_heroes(self) -> None:
        ally, enemy = _sample_draft()
        enemy.append(_pick("c1001", 7))

        result = fbr.recommend_final_bans(
            ally,
            enemy,
            ally_preban=["c9999"],
            first_pick_team="My Team",
            valid_heroes={str(p["hero"]) for p in ally} | {str(p["hero"]) for p in enemy} | {"c9999"},
            top_k=5,
        )

        self.assertNotIn("c1001", result["top_10_heroes"])
        self.assertNotIn("c9999", result["top_10_heroes"])
        self.assertTrue(all(hero.startswith("c2") for hero in result["top_10_heroes"]))

    def test_protected_enemy_hero_still_contributes_synergy_context(self) -> None:
        ally, enemy = _sample_draft()
        _set_lookups(
            synergy={
                ("c2005", "c2003"): fbr.SynergyEntry(score=0.90, same_team_count=10),
                ("c2003", "c2005"): fbr.SynergyEntry(score=0.90, same_team_count=10),
            }
        )

        result = fbr.recommend_final_bans(ally, enemy, top_k=3)
        top = result["recommendations"][0]
        self.assertEqual(top["hero"], "c2005")
        self.assertGreater(top["enemy_synergy_core_score"], 0.0)
        self.assertTrue(any(match["hero"] == "c2003" for match in top["debug"]["synergy_matches"]))

    def test_strong_ally_response_lowers_ban_score(self) -> None:
        ally, enemy = _sample_draft()
        _set_lookups(
            response={
                ("10", "c1005", "c2005"): fbr.ResponseEntry(score=0.95, response_count=20),
                ("8_9", "c1004", "c2005"): fbr.ResponseEntry(score=0.80, response_count=15),
            },
            response_evidence={"c2005", "c2004"},
        )

        high_response = fbr.recommend_final_bans(ally, enemy, top_k=4)
        low_response = fbr.recommend_final_bans(
            ally,
            enemy,
            top_k=4,
            response_lookup={},
            response_evidence={"c2005", "c2004"},
        )

        scored_high = next(item for item in high_response["recommendations"] if item["hero"] == "c2005")
        scored_low = next(item for item in low_response["recommendations"] if item["hero"] == "c2005")
        self.assertLess(scored_high["ally_lack_response_score"], scored_low["ally_lack_response_score"])

    def test_no_response_evidence_uses_neutral_lack_score(self) -> None:
        ally, enemy = _sample_draft()
        _set_lookups(response={}, response_evidence=set())

        result = fbr.recommend_final_bans(ally, enemy, top_k=3)
        for item in result["recommendations"]:
            self.assertEqual(item["ally_lack_response_score"], 0.50)

    def test_derive_ordered_picks_respects_first_pick_side(self) -> None:
        ally = fbr.derive_ordered_picks(
            ["c1001", "c1002", "c1003", "c1004", "c1005"],
            first_pick_team="My Team",
            side="ally",
        )
        enemy = fbr.derive_ordered_picks(
            ["c2001", "c2002", "c2003", "c2004", "c2005"],
            first_pick_team="My Team",
            side="enemy",
        )

        self.assertEqual([pick["order"] for pick in ally], [1, 4, 5, 8, 9])
        self.assertEqual([pick["order"] for pick in enemy], [2, 3, 6, 7, 10])

    def test_returns_all_four_bannable_enemy_picks_by_default(self) -> None:
        ally, enemy = _sample_draft()
        result = fbr.recommend_final_bans(ally, enemy, first_pick_team="My Team")
        self.assertEqual(len(result["top_10_heroes"]), 4)
        self.assertEqual(len(result["recommendations"]), 4)

    def test_enemy_first_pick_protects_order_5_only(self) -> None:
        ally = [
            _pick("c1001", 2),
            _pick("c1002", 3),
            _pick("c1003", 6),
            _pick("c1004", 7),
            _pick("c1005", 10),
        ]
        enemy = [
            _pick("c2001", 1),
            _pick("c2002", 4),
            _pick("c2003", 5),
            _pick("c2004", 8),
            _pick("c2005", 9),
        ]

        candidates, filtered_out = fbr.build_ban_candidates(
            ally_picks=ally,
            enemy_picks=enemy,
            first_pick_team="Enemy Team",
            valid_heroes={str(p["hero"]) for p in ally + enemy},
        )

        self.assertEqual({str(pick["hero"]) for pick in candidates}, {"c2001", "c2002", "c2004", "c2005"})
        self.assertTrue(
            any(item["hero"] == "c2003" and item["reason"] == "protected_order" for item in filtered_out)
        )

    def test_deterministic_ordering_on_ties(self) -> None:
        ally, enemy = _sample_draft()
        first = fbr.recommend_final_bans(ally, enemy, top_k=3)
        second = fbr.recommend_final_bans(ally, enemy, top_k=3)

        self.assertEqual(first["top_10_heroes"], second["top_10_heroes"])
        self.assertEqual(first["recommendations"][0]["hero"], "c2005")

    def test_top_rates_are_normalized_ban_score_shares(self) -> None:
        ally, enemy = _sample_draft()
        _set_lookups(
            synergy={
                ("c2005", "c2003"): fbr.SynergyEntry(score=0.90, same_team_count=10),
                ("c2003", "c2005"): fbr.SynergyEntry(score=0.90, same_team_count=10),
            }
        )

        result = fbr.recommend_final_bans(ally, enemy, top_k=2)
        self.assertEqual(len(result["top_10_rates"]), 2)
        self.assertAlmostEqual(sum(result["top_10_rates"]), 100.0)

    def test_top_rates_are_sharpened_for_display_contrast(self) -> None:
        recommendations = [
            {"ban_score": 0.50},
            {"ban_score": 0.40},
            {"ban_score": 0.30},
            {"ban_score": 0.20},
        ]

        rates = fbr._score_to_display_rates(recommendations)
        raw_gap = (0.50 / 1.40 * 100.0) - (0.40 / 1.40 * 100.0)

        self.assertAlmostEqual(sum(rates), 100.0)
        self.assertGreater(rates[0] - rates[1], raw_gap)

    def test_identical_scores_are_rank_weighted_for_display(self) -> None:
        recommendations = [{"ban_score": 0.5} for _ in range(4)]
        rates = fbr._score_to_display_rates(recommendations)

        self.assertAlmostEqual(sum(rates), 100.0)
        self.assertGreater(rates[0], rates[1])
        self.assertGreater(rates[1], rates[2])
        self.assertGreater(rates[2], rates[3])
        self.assertGreaterEqual(rates[0] - rates[3], 15.0)

    def test_zero_scores_return_zero_rates(self) -> None:
        recommendations = [{"ban_score": 0.0} for _ in range(4)]
        self.assertEqual(fbr._score_to_display_rates(recommendations), [0.0, 0.0, 0.0, 0.0])

    def test_hybrid_history_can_change_top_recommendation(self) -> None:
        ally, enemy = _sample_draft()
        artifact = {
            "has_labeled_decisions": True,
            "handled_by": "final_ban_hybrid_v2",
            "hybrid_config": {
                "prior_strength": 10.0,
                "confidence_strength": 5.0,
                "max_history_weight": 0.70,
            },
            "totals": {"ban_count": 20, "eligible_count": 80},
            "levels": {
                "level_1": {},
                "level_2": {},
                "level_3": {
                    "first|c2001": {"ban_count": 20, "eligible_count": 20},
                    "first|c2005": {"ban_count": 1, "eligible_count": 20},
                },
                "level_4": {
                    "c2001": {"ban_count": 20, "eligible_count": 20},
                    "c2005": {"ban_count": 1, "eligible_count": 20},
                },
            },
        }

        formula_only = fbr.recommend_final_bans(
            ally,
            enemy,
            top_k=4,
            final_ban_stats={"has_labeled_decisions": False},
        )
        hybrid = fbr.recommend_final_bans(
            ally,
            enemy,
            top_k=4,
            warfare_rules="Defense",
            final_ban_stats=artifact,
        )

        self.assertEqual(formula_only["handled_by"], "final_ban_stats_v1")
        self.assertEqual(hybrid["handled_by"], "final_ban_hybrid_v2")
        self.assertEqual(hybrid["recommendations"][0]["hero"], "c2001")
        top = hybrid["recommendations"][0]
        self.assertGreater(top["historical_weight"], 0.0)
        self.assertIn("formula_score", top)
        self.assertIn("historical_ban_rate", top)

    def test_missing_artifact_keeps_formula_fallback(self) -> None:
        ally, enemy = _sample_draft()
        with mock.patch.object(fbr, "load_final_ban_stats_artifact", return_value=None):
            fbr.reset_cached_stats()
            result = fbr.recommend_final_bans(ally, enemy, top_k=3)
        self.assertEqual(result["handled_by"], "final_ban_stats_v1")
        for item in result["recommendations"]:
            self.assertEqual(item["historical_weight"], 0.0)
            self.assertIsNone(item["historical_context_level"])


if __name__ == "__main__":
    unittest.main()
