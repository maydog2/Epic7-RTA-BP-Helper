"""Time-split evaluation for formula-only vs hybrid final-ban scoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.final_ban_recommender import (  # noqa: E402
    derive_ordered_picks,
    recommend_final_bans,
    reset_cached_stats,
)
from backend.runtime_paths import FINAL_BAN_EVALUATION_PATH, RAW_MATCH_HISTORY_PATH  # noqa: E402
from final_ban_stats_core import (  # noqa: E402
    DEFAULT_HYBRID_CONFIG,
    actor_pick_order_for_side,
    build_final_ban_stats_from_matches,
    lookup_historical_ban_rate,
)
from generate_final_ban_stats import load_matches  # noqa: E402
from match_history_utils import (  # noqa: E402
    ALLY_FINAL_BAN_TARGET,
    enemy_protected_order_for_match,
    enrich_match_draft,
    has_complete_final_ban_targets,
    validate_match_record,
)
from transformer_draft_data import extract_concrete_warfare_rule  # noqa: E402

HYBRID_PARAM_GRID = [
    {"prior_strength": 10.0, "confidence_strength": 20.0, "max_history_weight": 0.70},
    {"prior_strength": 8.0, "confidence_strength": 16.0, "max_history_weight": 0.70},
    {"prior_strength": 12.0, "confidence_strength": 24.0, "max_history_weight": 0.65},
]


def _decision_sort_key(match: dict[str, Any]) -> tuple:
    played_at = match.get("played_at")
    if played_at:
        return (0, str(played_at), int(match.get("match_id") or 0))
    return (1, str(match.get("match_id") or 0))


def labeled_ally_ban_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    for match in matches:
        if validate_match_record(match):
            continue
        if not has_complete_final_ban_targets(match):
            continue
        labeled.append(enrich_match_draft(match))
    return labeled


def split_matches(matches: list[dict[str, Any]], train_ratio: float = 0.8) -> tuple[list[dict], list[dict]]:
    ordered = sorted(matches, key=_decision_sort_key)
    if not ordered:
        return [], []
    split_index = max(1, int(len(ordered) * train_ratio))
    if split_index >= len(ordered):
        split_index = len(ordered) - 1
    return ordered[:split_index], ordered[split_index:]


def ally_ban_decision_payload(match: dict[str, Any]) -> dict[str, Any] | None:
    fps = match["first_pick_side"]
    draft = match["draft"]
    preban_set = set(match.get("ally_preban") or []) | set(match.get("enemy_preban") or [])
    protected = enemy_protected_order_for_match(fps)
    ally_picks = [entry for entry in draft if entry.get("side") == "ally"]
    enemy_picks = [entry for entry in draft if entry.get("side") == "enemy"]
    target = match.get(ALLY_FINAL_BAN_TARGET)
    if not target:
        return None

    first_pick_team = "My Team" if fps == "ally" else "Enemy Team"
    return {
        "match_id": match.get("match_id"),
        "first_pick_team": first_pick_team,
        "warfare_rules": extract_concrete_warfare_rule(match) or "ANY",
        "ally_picks": ally_picks,
        "enemy_picks": enemy_picks,
        "ally_preban": list(match.get("ally_preban") or []),
        "enemy_preban": list(match.get("enemy_preban") or []),
        "target": target,
        "protected_order": protected,
        "preban_set": preban_set,
    }


def _rank_target(
    *,
    payload: dict[str, Any],
    artifact: dict[str, Any] | None,
    use_hybrid: bool,
    hybrid_config: dict[str, float] | None = None,
) -> tuple[str | None, list[str]]:
    reset_cached_stats()
    result = recommend_final_bans(
        payload["ally_picks"],
        payload["enemy_picks"],
        ally_preban=payload["ally_preban"],
        enemy_preban=payload["enemy_preban"],
        first_pick_team=payload["first_pick_team"],
        warfare_rules=payload["warfare_rules"],
        top_k=4,
        final_ban_stats=artifact if use_hybrid else {"has_labeled_decisions": False},
    )
    ranked = [str(hero) for hero in result.get("top_10_heroes") or []]
    top1 = ranked[0] if ranked else None
    return top1, ranked


def _mrr(target: str, ranked: list[str]) -> float:
    for index, hero in enumerate(ranked, start=1):
        if hero == target:
            return 1.0 / index
    return 0.0


def _ndcg_at_k(target: str, ranked: list[str], k: int = 4) -> float:
    dcg = 0.0
    for index, hero in enumerate(ranked[:k], start=1):
        rel = 1.0 if hero == target else 0.0
        if rel:
            dcg += rel / __import__("math").log2(index + 1)
    idcg = 1.0 / __import__("math").log2(2)
    return dcg / idcg if idcg else 0.0


def evaluate_split(
    train_matches: list[dict[str, Any]],
    test_matches: list[dict[str, Any]],
    *,
    hybrid_config: dict[str, float] | None = None,
) -> dict[str, Any]:
    artifact = build_final_ban_stats_from_matches(train_matches)
    if hybrid_config:
        artifact["hybrid_config"] = dict(hybrid_config)

    formula_top1 = 0
    hybrid_top1 = 0
    formula_mrr = 0.0
    hybrid_mrr = 0.0
    formula_ndcg = 0.0
    hybrid_ndcg = 0.0
    evaluated = 0
    context_counts: dict[str, int] = {}

    for match in test_matches:
        payload = ally_ban_decision_payload(match)
        if payload is None:
            continue
        target = str(payload["target"])
        formula_top, formula_ranked = _rank_target(payload=payload, artifact=None, use_hybrid=False)
        hybrid_top, hybrid_ranked = _rank_target(
            payload=payload,
            artifact=artifact,
            use_hybrid=True,
            hybrid_config=hybrid_config,
        )
        evaluated += 1
        if formula_top == target:
            formula_top1 += 1
        if hybrid_top == target:
            hybrid_top1 += 1
        formula_mrr += _mrr(target, formula_ranked)
        hybrid_mrr += _mrr(target, hybrid_ranked)
        formula_ndcg += _ndcg_at_k(target, formula_ranked)
        hybrid_ndcg += _ndcg_at_k(target, hybrid_ranked)

        actor_order = actor_pick_order_for_side(match["first_pick_side"], "ally")
        for entry in payload["enemy_picks"]:
            if int(entry["order"]) == payload["protected_order"]:
                continue
            hero = str(entry["hero"])
            if hero in payload["preban_set"]:
                continue
            lookup = lookup_historical_ban_rate(
                artifact,
                actor_pick_order=actor_order,
                warfare_rule=payload["warfare_rules"],
                position_bucket=str(entry.get("position_bucket")),
                hero=hero,
                hybrid_config=artifact.get("hybrid_config"),
            )
            if lookup.context_level:
                context_counts[lookup.context_level] = context_counts.get(lookup.context_level, 0) + 1

    def _avg(value: float) -> float:
        return value / evaluated if evaluated else 0.0

    return {
        "evaluated_decisions": evaluated,
        "formula": {
            "top1_accuracy": _avg(formula_top1),
            "mrr": _avg(formula_mrr),
            "ndcg_at_4": _avg(formula_ndcg),
        },
        "hybrid": {
            "top1_accuracy": _avg(hybrid_top1),
            "mrr": _avg(hybrid_mrr),
            "ndcg_at_4": _avg(hybrid_ndcg),
        },
        "context_level_hits": context_counts,
        "train_decision_count": artifact.get("decision_count", 0),
    }


def choose_hybrid_config(train_matches: list[dict[str, Any]], test_matches: list[dict[str, Any]]) -> dict[str, float]:
    best = dict(DEFAULT_HYBRID_CONFIG)
    best_score = -1.0
    for candidate in HYBRID_PARAM_GRID:
        metrics = evaluate_split(train_matches, test_matches, hybrid_config=candidate)
        score = metrics["hybrid"]["mrr"]
        if score > best_score:
            best_score = score
            best = dict(candidate)
    return best


def run_evaluation(*, raw_path: Path = RAW_MATCH_HISTORY_PATH) -> dict[str, Any]:
    matches = load_matches(raw_path) if raw_path.exists() else []
    labeled = labeled_ally_ban_matches(matches)
    split_method = "played_at_or_match_id"
    train_matches, test_matches = split_matches(labeled)

    insufficient = len(labeled) < 20 or len(test_matches) < 4
    report: dict[str, Any] = {
        "raw_path": str(raw_path),
        "labeled_match_count": len(labeled),
        "train_match_count": len(train_matches),
        "test_match_count": len(test_matches),
        "split_method": split_method,
        "insufficient_data": insufficient,
        "selected_hybrid_config": dict(DEFAULT_HYBRID_CONFIG),
        "metrics": None,
    }

    if insufficient:
        report["note"] = (
            "Insufficient labeled final-ban data for reliable holdout evaluation. "
            "Re-scrape with E7_GET_MATCHES_REVISIT=1 to backfill ally/enemy final-ban targets."
        )
        FINAL_BAN_EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        FINAL_BAN_EVALUATION_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    selected = choose_hybrid_config(train_matches, test_matches)
    metrics = evaluate_split(train_matches, test_matches, hybrid_config=selected)
    report["selected_hybrid_config"] = selected
    report["metrics"] = metrics
    FINAL_BAN_EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_BAN_EVALUATION_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate formula-only vs hybrid final-ban scoring.")
    parser.add_argument("--raw-path", type=Path, default=RAW_MATCH_HISTORY_PATH)
    args = parser.parse_args()
    report = run_evaluation(raw_path=args.raw_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
