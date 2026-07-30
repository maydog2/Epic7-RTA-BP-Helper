import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys

WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "workflow_scripts"
if str(WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_DIR))

from final_ban_stats_core import (  # noqa: E402
    actor_pick_order_for_side,
    build_final_ban_stats_from_matches,
    lookup_historical_ban_rate,
)
from generate_final_ban_stats import refresh_final_ban_stats  # noqa: E402
from match_history_utils import make_draft_entry  # noqa: E402


def _labeled_match(
    *,
    match_id: int,
    first_pick_side: str = "ally",
    ally_target: str = "c2002",
    enemy_target: str = "c1005",
    warfare_rules: str = "Defense",
) -> dict:
    if first_pick_side == "ally":
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
    else:
        draft = [
            make_draft_entry(order=1, side="enemy", hero="c2001"),
            make_draft_entry(order=2, side="ally", hero="c1001"),
            make_draft_entry(order=3, side="ally", hero="c1002"),
            make_draft_entry(order=4, side="enemy", hero="c2002"),
            make_draft_entry(order=5, side="enemy", hero="c2003"),
            make_draft_entry(order=6, side="ally", hero="c1003"),
            make_draft_entry(order=7, side="ally", hero="c1004"),
            make_draft_entry(order=8, side="enemy", hero="c2004"),
            make_draft_entry(order=9, side="enemy", hero="c2005"),
            make_draft_entry(order=10, side="ally", hero="c1005"),
        ]

    return {
        "match_id": match_id,
        "first_pick_side": first_pick_side,
        "winner_side": first_pick_side,
        "warfare_rules": warfare_rules,
        "ally_preban": ["c9001", "c9002"],
        "enemy_preban": ["c9101", "c9102"],
        "ally_final_ban_target": ally_target,
        "enemy_final_ban_target": enemy_target,
        "draft": draft,
    }


class GenerateFinalBanStatsTests(unittest.TestCase):
    def test_builds_eligible_and_ban_counts_for_both_decisions(self) -> None:
        matches = [_labeled_match(match_id=1), _labeled_match(match_id=2, ally_target="c2004")]
        artifact = build_final_ban_stats_from_matches(matches)

        self.assertEqual(artifact["decision_count"], 4)
        self.assertEqual(artifact["labeled_match_count"], 2)
        self.assertTrue(artifact["has_labeled_decisions"])
        self.assertEqual(artifact["handled_by"], "final_ban_hybrid_v2")

        level2 = artifact["levels"]["level_2"]
        ally_first_key = "first|2_3|c2002"
        self.assertGreaterEqual(level2[ally_first_key]["eligible_count"], 2)
        self.assertGreaterEqual(level2[ally_first_key]["ban_count"], 1)

    def test_protected_pick_is_not_counted_as_eligible(self) -> None:
        matches = [_labeled_match(match_id=1, ally_target="c2003")]
        artifact = build_final_ban_stats_from_matches(matches)
        level3 = artifact["levels"]["level_3"]
        protected_key = "first|c2003"
        protected_counts = level3.get(protected_key, {"eligible_count": 0, "ban_count": 0})
        self.assertEqual(protected_counts["eligible_count"], 0)

    def test_missing_labels_produce_empty_artifact(self) -> None:
        match = _labeled_match(match_id=1)
        del match["ally_final_ban_target"]
        del match["enemy_final_ban_target"]
        artifact = build_final_ban_stats_from_matches([match])
        self.assertEqual(artifact["decision_count"], 0)
        self.assertFalse(artifact["has_labeled_decisions"])

    def test_lookup_falls_back_from_rule_to_actor_bucket(self) -> None:
        matches = [
            _labeled_match(match_id=index, warfare_rules="Defense", ally_target="c2002")
            for index in range(1, 6)
        ]
        artifact = build_final_ban_stats_from_matches(matches)
        lookup = lookup_historical_ban_rate(
            artifact,
            actor_pick_order="first",
            warfare_rule="Resistance",
            position_bucket="2_3",
            hero="c2002",
        )
        self.assertEqual(lookup.context_level, "actor_bucket_hero")
        self.assertGreater(lookup.eligible_count, 0)

    def test_refresh_writes_runtime_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = Path(tmp_dir) / "raw.jsonl"
            raw_path.write_text(
                json.dumps(_labeled_match(match_id=1)) + "\n",
                encoding="utf-8",
            )
            runtime_dir = Path(tmp_dir) / "runtime"
            runtime_dir.mkdir()
            with mock.patch("generate_final_ban_stats.RUNTIME_DATA_DIR", runtime_dir), mock.patch(
                "generate_final_ban_stats.FINAL_BAN_STATS_PATH",
                runtime_dir / "final_ban_stats.pkl",
            ):
                summary = refresh_final_ban_stats(raw_path=raw_path)
                self.assertTrue((runtime_dir / "final_ban_stats.pkl").exists())
                self.assertEqual(summary["artifact"]["decision_count"], 2)


if __name__ == "__main__":
    unittest.main()
