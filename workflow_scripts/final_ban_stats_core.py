"""Shared final-ban historical stats structures, builders, and hybrid lookup."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from match_history_utils import (
    ALLY_FINAL_BAN_TARGET,
    ENEMY_FINAL_BAN_TARGET,
    ally_protected_order_for_match,
    enemy_protected_order_for_match,
    enrich_match_draft,
    get_position_bucket,
    has_complete_final_ban_targets,
    validate_match_record,
)
from transformer_draft_data import extract_concrete_warfare_rule

FINAL_BAN_STATS_VERSION = 1
HANDLED_BY_V1 = "final_ban_stats_v1"
HANDLED_BY_HYBRID = "final_ban_hybrid_v2"

DEFAULT_HYBRID_CONFIG: dict[str, float] = {
    "prior_strength": 10.0,
    "confidence_strength": 20.0,
    "max_history_weight": 0.70,
}

LEVEL_NAMES = ("level_1", "level_2", "level_3", "level_4")
CONTEXT_LEVEL_LABELS = {
    "level_1": "actor_rule_bucket_hero",
    "level_2": "actor_bucket_hero",
    "level_3": "actor_hero",
    "level_4": "hero_global",
}


@dataclass(frozen=True)
class HistoricalLookup:
    historical_score: float
    ban_count: int
    eligible_count: int
    context_level: str | None
    confidence: float
    history_weight: float
    parent_rate: float


def actor_pick_order_for_side(first_pick_side: str, actor_side: str) -> str:
    if actor_side not in {"ally", "enemy"}:
        raise ValueError(f"invalid actor_side: {actor_side}")
    if first_pick_side not in {"ally", "enemy"}:
        raise ValueError(f"invalid first_pick_side: {first_pick_side}")
    is_first = first_pick_side == actor_side
    return "first" if is_first else "second"


def actor_pick_order_for_ui_first_pick_team(first_pick_team: str) -> str:
    return "first" if first_pick_team == "My Team" else "second"


def encode_level_key(parts: tuple[str, ...] | str) -> str:
    if isinstance(parts, str):
        return parts
    return "|".join(parts)


def decode_level_key(key: str) -> tuple[str, ...]:
    return tuple(key.split("|"))


def empty_count_map() -> dict[str, dict[str, int]]:
    return {"ban_count": 0, "eligible_count": 0}


def increment_count(store: dict[str, dict[str, int]], key: str, *, ban: bool = False) -> None:
    entry = store.setdefault(key, empty_count_map())
    entry["eligible_count"] += 1
    if ban:
        entry["ban_count"] += 1


def _eligible_ban_candidates(
    draft: list[dict[str, Any]],
    *,
    target_side: str,
    protected_order: int,
    preban_set: set[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for entry in draft:
        if entry.get("side") != target_side:
            continue
        hero = entry.get("hero")
        order = int(entry.get("order") or 0)
        if order == protected_order:
            continue
        if not hero or hero in preban_set:
            continue
        candidates.append(entry)
    return candidates


def accumulate_decision(
    levels: dict[str, dict[str, dict[str, int]]],
    *,
    actor_pick_order: str,
    warfare_rule: str | None,
    candidates: list[dict[str, Any]],
    ban_target: str | None,
) -> None:
    for entry in candidates:
        hero = str(entry["hero"])
        bucket = str(entry.get("position_bucket") or get_position_bucket(int(entry["order"])))
        is_ban = hero == ban_target

        if warfare_rule:
            key1 = encode_level_key((actor_pick_order, warfare_rule, bucket, hero))
            increment_count(levels["level_1"], key1, ban=is_ban)

        key2 = encode_level_key((actor_pick_order, bucket, hero))
        increment_count(levels["level_2"], key2, ban=is_ban)

        key3 = encode_level_key((actor_pick_order, hero))
        increment_count(levels["level_3"], key3, ban=is_ban)

        increment_count(levels["level_4"], hero, ban=is_ban)


def build_empty_artifact(*, skip_reasons: dict[str, int] | None = None) -> dict[str, Any]:
    return {
        "version": FINAL_BAN_STATS_VERSION,
        "decision_count": 0,
        "labeled_match_count": 0,
        "match_count_seen": 0,
        "match_count_with_labels": 0,
        "skip_reasons": dict(skip_reasons or {}),
        "hybrid_config": dict(DEFAULT_HYBRID_CONFIG),
        "levels": {level: {} for level in LEVEL_NAMES},
        "totals": {"ban_count": 0, "eligible_count": 0},
        "handled_by": HANDLED_BY_V1,
        "has_labeled_decisions": False,
    }


def build_final_ban_stats_from_matches(matches: list[dict[str, Any]]) -> dict[str, Any]:
    levels: dict[str, dict[str, dict[str, int]]] = {level: {} for level in LEVEL_NAMES}
    skip_reasons: dict[str, int] = defaultdict(int)
    labeled_match_count = 0
    decision_count = 0
    match_count_seen = 0

    for match in matches:
        match_count_seen += 1
        errors = validate_match_record(match)
        if errors:
            skip_reasons["invalid_match"] += 1
            continue

        if not has_complete_final_ban_targets(match):
            skip_reasons["skipped_missing_final_ban"] += 1
            continue

        draft = enrich_match_draft(match)["draft"]
        fps = match["first_pick_side"]
        preban_set = set(match.get("ally_preban") or []) | set(match.get("enemy_preban") or [])
        warfare_rule = extract_concrete_warfare_rule(match)

        ally_target = match.get(ALLY_FINAL_BAN_TARGET)
        enemy_target = match.get(ENEMY_FINAL_BAN_TARGET)
        if not ally_target or not enemy_target:
            skip_reasons["skipped_missing_final_ban"] += 1
            continue

        labeled_match_count += 1

        ally_candidates = _eligible_ban_candidates(
            draft,
            target_side="enemy",
            protected_order=enemy_protected_order_for_match(fps),
            preban_set=preban_set,
        )
        if ally_target not in {str(entry["hero"]) for entry in ally_candidates}:
            skip_reasons["invalid_ally_ban_target"] += 1
        else:
            accumulate_decision(
                levels,
                actor_pick_order=actor_pick_order_for_side(fps, "ally"),
                warfare_rule=warfare_rule,
                candidates=ally_candidates,
                ban_target=str(ally_target),
            )
            decision_count += 1

        enemy_candidates = _eligible_ban_candidates(
            draft,
            target_side="ally",
            protected_order=ally_protected_order_for_match(fps),
            preban_set=preban_set,
        )
        if enemy_target not in {str(entry["hero"]) for entry in enemy_candidates}:
            skip_reasons["invalid_enemy_ban_target"] += 1
        else:
            accumulate_decision(
                levels,
                actor_pick_order=actor_pick_order_for_side(fps, "enemy"),
                warfare_rule=warfare_rule,
                candidates=enemy_candidates,
                ban_target=str(enemy_target),
            )
            decision_count += 1

    totals = {"ban_count": 0, "eligible_count": 0}
    for level_store in levels.values():
        for entry in level_store.values():
            totals["ban_count"] += int(entry.get("ban_count", 0))
            totals["eligible_count"] += int(entry.get("eligible_count", 0))

    has_labeled = decision_count > 0
    return {
        "version": FINAL_BAN_STATS_VERSION,
        "decision_count": decision_count,
        "labeled_match_count": labeled_match_count,
        "match_count_seen": match_count_seen,
        "match_count_with_labels": labeled_match_count,
        "skip_reasons": dict(skip_reasons),
        "hybrid_config": dict(DEFAULT_HYBRID_CONFIG),
        "levels": levels,
        "totals": totals,
        "handled_by": HANDLED_BY_HYBRID if has_labeled else HANDLED_BY_V1,
        "has_labeled_decisions": has_labeled,
    }


def global_baseline_rate(artifact: dict[str, Any]) -> float:
    totals = artifact.get("totals") or {}
    eligible = int(totals.get("eligible_count") or 0)
    if eligible <= 0:
        return 0.25
    return int(totals.get("ban_count") or 0) / eligible


def _entry_counts(level_store: dict[str, dict[str, int]], key: str) -> tuple[int, int]:
    entry = level_store.get(key) or {}
    return int(entry.get("ban_count") or 0), int(entry.get("eligible_count") or 0)


def _smoothed_rate(ban_count: int, eligible_count: int, parent_rate: float, prior_strength: float) -> float:
    return (ban_count + prior_strength * parent_rate) / (eligible_count + prior_strength)


def _parent_rate_for_level(
    artifact: dict[str, Any],
    *,
    level_name: str,
    actor_pick_order: str,
    warfare_rule: str | None,
    position_bucket: str,
    hero: str,
    prior_strength: float,
    baseline: float,
) -> float:
    levels = artifact["levels"]

    if level_name == "level_1":
        key3 = encode_level_key((actor_pick_order, hero))
        bc, ec = _entry_counts(levels["level_3"], key3)
        if ec > 0:
            key4_bc, key4_ec = _entry_counts(levels["level_4"], hero)
            parent = _smoothed_rate(key4_bc, key4_ec, baseline, prior_strength) if key4_ec > 0 else baseline
            return _smoothed_rate(bc, ec, parent, prior_strength)
        key4_bc, key4_ec = _entry_counts(levels["level_4"], hero)
        if key4_ec > 0:
            return _smoothed_rate(key4_bc, key4_ec, baseline, prior_strength)
        return baseline

    if level_name == "level_2":
        key3 = encode_level_key((actor_pick_order, hero))
        bc, ec = _entry_counts(levels["level_3"], key3)
        if ec > 0:
            key4_bc, key4_ec = _entry_counts(levels["level_4"], hero)
            parent = _smoothed_rate(key4_bc, key4_ec, baseline, prior_strength) if key4_ec > 0 else baseline
            return _smoothed_rate(bc, ec, parent, prior_strength)
        key4_bc, key4_ec = _entry_counts(levels["level_4"], hero)
        if key4_ec > 0:
            return _smoothed_rate(key4_bc, key4_ec, baseline, prior_strength)
        return baseline

    if level_name == "level_3":
        bc, ec = _entry_counts(levels["level_4"], hero)
        if ec > 0:
            return _smoothed_rate(bc, ec, baseline, prior_strength)
        return baseline

    return baseline


def lookup_historical_ban_rate(
    artifact: dict[str, Any] | None,
    *,
    actor_pick_order: str,
    warfare_rule: str | None,
    position_bucket: str,
    hero: str,
    hybrid_config: dict[str, float] | None = None,
) -> HistoricalLookup:
    config = hybrid_config or (artifact or {}).get("hybrid_config") or DEFAULT_HYBRID_CONFIG
    prior_strength = float(config.get("prior_strength", DEFAULT_HYBRID_CONFIG["prior_strength"]))
    confidence_strength = float(config.get("confidence_strength", DEFAULT_HYBRID_CONFIG["confidence_strength"]))
    max_history_weight = float(config.get("max_history_weight", DEFAULT_HYBRID_CONFIG["max_history_weight"]))

    if not artifact or not artifact.get("has_labeled_decisions"):
        return HistoricalLookup(0.0, 0, 0, None, 0.0, 0.0, 0.25)

    levels = artifact["levels"]
    baseline = global_baseline_rate(artifact)
    normalized_rule = None if not warfare_rule or warfare_rule.upper() == "ANY" else warfare_rule

    lookup_chain: list[tuple[str, str]] = []
    if normalized_rule:
        lookup_chain.append(
            (
                "level_1",
                encode_level_key((actor_pick_order, normalized_rule, position_bucket, hero)),
            )
        )
    lookup_chain.extend(
        [
            ("level_2", encode_level_key((actor_pick_order, position_bucket, hero))),
            ("level_3", encode_level_key((actor_pick_order, hero))),
            ("level_4", hero),
        ]
    )

    for level_name, key in lookup_chain:
        ban_count, eligible_count = _entry_counts(levels[level_name], key)
        if eligible_count <= 0:
            continue
        parent_rate = _parent_rate_for_level(
            artifact,
            level_name=level_name,
            actor_pick_order=actor_pick_order,
            warfare_rule=normalized_rule,
            position_bucket=position_bucket,
            hero=hero,
            prior_strength=prior_strength,
            baseline=baseline,
        )
        historical_score = _smoothed_rate(ban_count, eligible_count, parent_rate, prior_strength)
        confidence = eligible_count / (eligible_count + confidence_strength)
        history_weight = max_history_weight * confidence
        return HistoricalLookup(
            historical_score=min(max(historical_score, 0.0), 1.0),
            ban_count=ban_count,
            eligible_count=eligible_count,
            context_level=CONTEXT_LEVEL_LABELS[level_name],
            confidence=confidence,
            history_weight=history_weight,
            parent_rate=parent_rate,
        )

    return HistoricalLookup(0.0, 0, 0, None, 0.0, 0.0, baseline)


def blend_hybrid_score(formula_score: float, lookup: HistoricalLookup) -> tuple[float, float]:
    weight = lookup.history_weight
    final_score = weight * lookup.historical_score + (1.0 - weight) * formula_score
    return min(max(final_score, 0.0), 1.0), weight


def export_levels_for_json(levels: dict[str, dict[str, dict[str, int]]]) -> dict[str, dict[str, dict[str, int]]]:
    exported: dict[str, dict[str, dict[str, int]]] = {}
    for level_name, store in levels.items():
        exported[level_name] = {
            key: {
                "ban_count": int(entry.get("ban_count") or 0),
                "eligible_count": int(entry.get("eligible_count") or 0),
            }
            for key, entry in store.items()
        }
    return exported
