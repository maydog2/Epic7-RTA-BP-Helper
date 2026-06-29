"""Explainable final-ban recommendations from frequency-based draft statistics."""

from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from .runtime_paths import RERANKER_DATA_DIR, WORKFLOW_DIR

if str(WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_DIR))

from match_history_utils import get_position_bucket  # noqa: E402

SYNERGY_STATS_PATH = RERANKER_DATA_DIR / "hero_synergy_stats.csv"
RESPONSE_STATS_PATH = RERANKER_DATA_DIR / "hero_counter_or_response_stats.csv"

FIRST_PICK_ORDERS = [1, 4, 5, 8, 9]
SECOND_PICK_ORDERS = [2, 3, 6, 7, 10]
MAX_BANNABLE_ENEMY_PICKS = 4

MIN_SYNERGY_COUNT = 3
MIN_RESPONSE_COUNT = 3

WEIGHT_SYNERGY = 0.48
WEIGHT_LACK_RESPONSE = 0.40
WEIGHT_POSITION_THREAT = 0.12
BAN_RATE_EXPONENT = 3.0
BAN_RATE_RANK_WEIGHTS = (1.35, 1.0, 0.78, 0.62)
BANNABLE_SOLE_COUNTER_COVERAGE_MULTIPLIER = 0.25

SUPPORTED_RESPONSE_BUCKETS = frozenset({"2_3", "4", "7", "8_9", "10"})

LATE_RESPONSE_BUCKET_WEIGHTS: dict[str, float] = {
    "7": 0.20,
    "8_9": 0.40,
    "10": 0.40,
}

POSITION_BUCKET_RESPONSE_WEIGHTS: dict[str, dict[str, float]] = {
    "2_3": {"2_3": 1.0},
    "4": {"4": 1.0},
    "7": {"7": 1.0},
    "8_9": {"8_9": 1.0},
    "10": {"10": 1.0},
}

ORDER_THREAT_SCORES: dict[int, float] = {
    2: 0.45,
    3: 0.50,
    4: 0.55,
    7: 0.62,
    8: 0.72,
    9: 0.72,
    10: 0.78,
}

INVALID_HERO_TOKENS = frozenset({"", "unknown", "<PAD>", "<UNK>"})
HANDLED_BY = "final_ban_stats_v1"


@dataclass(frozen=True)
class SynergyEntry:
    score: float
    same_team_count: int


@dataclass(frozen=True)
class ResponseEntry:
    score: float
    response_count: int


_synergy_lookup: dict[tuple[str, str], SynergyEntry] | None = None
_response_lookup: dict[tuple[str, str, str], ResponseEntry] | None = None
_candidates_with_response_evidence: set[str] | None = None


def reset_cached_stats() -> None:
    global _synergy_lookup, _response_lookup, _candidates_with_response_evidence
    _synergy_lookup = None
    _response_lookup = None
    _candidates_with_response_evidence = None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        logging.warning("Final-ban artifact not found: %s", path)
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_synergy_lookup() -> dict[tuple[str, str], SynergyEntry]:
    global _synergy_lookup
    if _synergy_lookup is not None:
        return _synergy_lookup

    lookup: dict[tuple[str, str], SynergyEntry] = {}
    for row in _read_csv_rows(SYNERGY_STATS_PATH):
        try:
            same_team_count = int(float(row["same_team_count"]))
            score = float(row["normalized_synergy_score"])
        except (KeyError, TypeError, ValueError):
            continue
        if same_team_count < MIN_SYNERGY_COUNT:
            continue
        hero_a = row["hero_a"]
        hero_b = row["hero_b"]
        entry = SynergyEntry(score=min(max(score, 0.0), 1.0), same_team_count=same_team_count)
        lookup[(hero_a, hero_b)] = entry
        lookup[(hero_b, hero_a)] = entry

    _synergy_lookup = lookup
    return lookup


def load_response_lookup() -> dict[tuple[str, str, str], ResponseEntry]:
    global _response_lookup, _candidates_with_response_evidence
    if _response_lookup is not None:
        return _response_lookup

    lookup: dict[tuple[str, str, str], ResponseEntry] = {}
    evidence: set[str] = set()
    for row in _read_csv_rows(RESPONSE_STATS_PATH):
        try:
            response_count = int(float(row["response_count"]))
            score = float(row["normalized_response_score"])
        except (KeyError, TypeError, ValueError):
            continue
        if response_count < MIN_RESPONSE_COUNT:
            continue
        bucket = row["position_bucket"]
        if bucket not in SUPPORTED_RESPONSE_BUCKETS:
            continue
        candidate = row["candidate_hero"]
        enemy = row["enemy_hero"]
        lookup[(bucket, candidate, enemy)] = ResponseEntry(
            score=min(max(score, 0.0), 1.0),
            response_count=response_count,
        )
        evidence.add(enemy)

    _response_lookup = lookup
    _candidates_with_response_evidence = evidence
    return lookup


def candidates_with_response_evidence() -> set[str]:
    if _candidates_with_response_evidence is None:
        load_response_lookup()
    return _candidates_with_response_evidence or set()


def is_valid_hero(hero: str | None, valid_heroes: set[str] | None = None) -> bool:
    if not hero or hero in INVALID_HERO_TOKENS:
        return False
    if valid_heroes is not None and hero not in valid_heroes:
        return False
    return True


def enemy_protected_order(first_pick_team: str) -> int:
    """Global draft order of the enemy team's protected (unbannable) pick."""
    return 5 if first_pick_team == "Enemy Team" else 6


def ally_protected_order(first_pick_team: str) -> int:
    """Global draft order of the ally team's protected (enemy-unbannable) pick."""
    return 5 if first_pick_team == "My Team" else 6


def derive_ordered_picks(
    hero_codes: list[str],
    *,
    first_pick_team: str,
    side: str,
) -> list[dict[str, object]]:
    """Map side-local pick order to global draft orders."""
    if side == "ally":
        is_first_pick = first_pick_team == "My Team"
    else:
        is_first_pick = first_pick_team == "Enemy Team"
    orders = FIRST_PICK_ORDERS if is_first_pick else SECOND_PICK_ORDERS
    picks: list[dict[str, object]] = []
    for index, hero in enumerate(hero_codes[:5]):
        order = orders[index]
        picks.append(
            {
                "hero": hero,
                "order": order,
                "position_bucket": get_position_bucket(order),
            }
        )
    return picks


def build_ban_candidates(
    *,
    ally_picks: list[dict[str, object]],
    enemy_picks: list[dict[str, object]],
    first_pick_team: str,
    ally_preban: list[str] | None = None,
    enemy_preban: list[str] | None = None,
    valid_heroes: set[str] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    protected_order = enemy_protected_order(first_pick_team)
    ally_hero_set = {
        str(pick["hero"])
        for pick in ally_picks
        if pick.get("hero") is not None
    }
    preban_set = set(ally_preban or []) | set(enemy_preban or [])

    candidates: list[dict[str, object]] = []
    filtered_out: list[dict[str, object]] = []

    for pick in enemy_picks:
        hero = str(pick.get("hero") or "")
        order = int(pick.get("order") or 0)
        reason = None

        if order == protected_order:
            reason = "protected_order"
        elif not is_valid_hero(hero, valid_heroes):
            reason = "invalid_hero"
        elif hero in ally_hero_set:
            reason = "ally_pick_overlap"
        elif hero in preban_set:
            reason = "prebanned"

        if reason:
            filtered_out.append({"hero": hero, "order": order, "reason": reason})
            continue

        candidates.append(dict(pick))

    return candidates, filtered_out


def _combine_synergy_scores(scores: list[float]) -> float:
    if not scores:
        return 0.0
    average = sum(scores) / len(scores)
    maximum = max(scores)
    return min(max(0.60 * average + 0.40 * maximum, 0.0), 1.0)


def enemy_synergy_core_score(
    candidate_hero: str,
    enemy_context: list[dict[str, object]],
    *,
    synergy_lookup: dict[tuple[str, str], SynergyEntry] | None = None,
) -> tuple[float, list[dict[str, object]]]:
    lookup = synergy_lookup if synergy_lookup is not None else load_synergy_lookup()
    matched_scores: list[float] = []
    matches: list[dict[str, object]] = []

    for context_pick in enemy_context:
        context_hero = str(context_pick.get("hero") or "")
        if not context_hero or context_hero == candidate_hero:
            continue
        entry = lookup.get((candidate_hero, context_hero))
        if entry is None:
            continue
        matched_scores.append(entry.score)
        matches.append(
            {
                "hero": context_hero,
                "order": context_pick.get("order"),
                "same_team_count": entry.same_team_count,
                "score": entry.score,
            }
        )

    return _combine_synergy_scores(matched_scores), matches


def response_bucket_weights_for_position(position_bucket: str | None) -> dict[str, float]:
    """Map enemy candidate position bucket to response CSV buckets."""
    if position_bucket in POSITION_BUCKET_RESPONSE_WEIGHTS:
        return POSITION_BUCKET_RESPONSE_WEIGHTS[position_bucket]
    return dict(LATE_RESPONSE_BUCKET_WEIGHTS)


def _ally_response_coverage(
    ally_hero: str,
    enemy_candidate: str,
    *,
    response_lookup: dict[tuple[str, str, str], ResponseEntry] | None = None,
    bucket_weights: dict[str, float] | None = None,
) -> tuple[float, list[dict[str, object]]]:
    lookup = response_lookup if response_lookup is not None else load_response_lookup()
    selected_buckets = bucket_weights if bucket_weights is not None else LATE_RESPONSE_BUCKET_WEIGHTS
    weighted_sum = 0.0
    weight_total = 0.0
    matches: list[dict[str, object]] = []

    for bucket, bucket_weight in selected_buckets.items():
        entry = lookup.get((bucket, ally_hero, enemy_candidate))
        if entry is None:
            continue
        weighted_sum += bucket_weight * entry.score
        weight_total += bucket_weight
        matches.append(
            {
                "ally_hero": ally_hero,
                "bucket": bucket,
                "response_count": entry.response_count,
                "score": entry.score,
            }
        )

    if weight_total <= 0.0:
        return 0.0, matches
    return min(weighted_sum / weight_total, 1.0), matches


def _combine_response_source_coverages(coverages: list[float]) -> float:
    if not coverages:
        return 0.0
    sorted_coverages = sorted(coverages, reverse=True)
    if len(sorted_coverages) == 1:
        return sorted_coverages[0]
    return min(1.0, 0.75 * sorted_coverages[0] + 0.25 * sorted_coverages[1])


def _apply_sole_bannable_counter_discount(
    response_sources: list[dict[str, object]],
    original_team_coverage: float,
) -> tuple[float, bool]:
    if len(response_sources) != 1:
        return original_team_coverage, False
    if bool(response_sources[0]["is_protected"]):
        return original_team_coverage, False
    effective_coverage = original_team_coverage * BANNABLE_SOLE_COUNTER_COVERAGE_MULTIPLIER
    return effective_coverage, True


def ally_lack_response_score(
    enemy_candidate: str,
    ally_picks: list[dict[str, object]],
    *,
    first_pick_team: str = "My Team",
    position_bucket: str | None = None,
    response_lookup: dict[tuple[str, str, str], ResponseEntry] | None = None,
    response_evidence: set[str] | None = None,
) -> tuple[float, list[dict[str, object]], float, list[str], dict[str, object]]:
    lookup = response_lookup if response_lookup is not None else load_response_lookup()
    evidence = response_evidence if response_evidence is not None else candidates_with_response_evidence()
    bucket_weights = response_bucket_weights_for_position(position_bucket)
    response_buckets_used = sorted(bucket_weights.keys())
    protected_order = ally_protected_order(first_pick_team)

    response_sources: list[dict[str, object]] = []
    all_matches: list[dict[str, object]] = []

    for pick in ally_picks:
        ally_hero = str(pick.get("hero") or "")
        if not is_valid_hero(ally_hero):
            continue
        order = int(pick.get("order") or 0)
        coverage, matches = _ally_response_coverage(
            ally_hero,
            enemy_candidate,
            response_lookup=lookup,
            bucket_weights=bucket_weights,
        )
        if not matches:
            continue
        response_sources.append(
            {
                "ally_hero": ally_hero,
                "order": order,
                "is_protected": order == protected_order,
                "coverage": round(coverage, 6),
            }
        )
        all_matches.extend(matches)

    response_debug: dict[str, object] = {
        "response_sources": response_sources,
        "sole_counter_is_bannable": False,
        "effective_team_response_coverage": 0.0,
        "original_team_response_coverage": 0.0,
    }

    if enemy_candidate not in evidence:
        return 0.50, all_matches, 0.0, response_buckets_used, response_debug

    if not response_sources:
        return 1.0, all_matches, 0.0, response_buckets_used, response_debug

    original_team_coverage = _combine_response_source_coverages(
        [float(source["coverage"]) for source in response_sources]
    )
    effective_team_coverage, sole_counter_is_bannable = _apply_sole_bannable_counter_discount(
        response_sources,
        original_team_coverage,
    )
    response_debug["sole_counter_is_bannable"] = sole_counter_is_bannable
    response_debug["original_team_response_coverage"] = round(original_team_coverage, 6)
    response_debug["effective_team_response_coverage"] = round(effective_team_coverage, 6)

    lack_score = min(max(1.0 - effective_team_coverage, 0.0), 1.0)
    return lack_score, all_matches, effective_team_coverage, response_buckets_used, response_debug


def enemy_position_threat_score(order: int | None) -> float:
    if order is None:
        return 0.50
    return ORDER_THREAT_SCORES.get(int(order), 0.50)


def _build_reasons(
    *,
    synergy_score: float,
    lack_score: float,
    threat_score: float,
    synergy_matches: list[dict[str, object]],
    response_matches: list[dict[str, object]],
    sole_counter_is_bannable: bool = False,
    order: int,
) -> list[str]:
    reasons: list[str] = []
    if synergy_score >= 0.45 and synergy_matches:
        reasons.append("Strong synergy with enemy team core")
    if lack_score >= 0.65:
        reasons.append("Ally team lacks known position-bucket responses")
    elif lack_score <= 0.35 and response_matches:
        reasons.append("Ally team already has response coverage")
    if sole_counter_is_bannable:
        reasons.append("Only known response is bannable")
    if threat_score >= 0.85:
        reasons.append(f"High late-pick threat at order {order}")
    elif threat_score >= 0.70:
        reasons.append(f"Mid/late draft threat at order {order}")
    if not reasons:
        reasons.append("Balanced ban value from draft-position and team-context stats")
    return reasons


def _sort_key(recommendation: dict[str, object]) -> tuple:
    return (
        -float(recommendation["ban_score"]),
        -float(recommendation["enemy_synergy_core_score"]),
        -float(recommendation["ally_lack_response_score"]),
        -float(recommendation["enemy_position_threat_score"]),
        int(recommendation["order"]),
        str(recommendation["hero"]),
    )


def _score_to_display_rates(recommendations: list[dict[str, object]]) -> list[float]:
    sharpened_scores = [
        (max(float(item["ban_score"]), 0.0) ** BAN_RATE_EXPONENT)
        * BAN_RATE_RANK_WEIGHTS[min(index, len(BAN_RATE_RANK_WEIGHTS) - 1)]
        for index, item in enumerate(recommendations)
    ]
    total_score = sum(sharpened_scores)
    if total_score <= 0.0:
        return [0.0 for _ in recommendations]
    return [(score / total_score) * 100.0 for score in sharpened_scores]


def recommend_final_bans(
    ally_picks: list[dict[str, object]],
    enemy_picks: list[dict[str, object]],
    ally_preban: list[str] | None = None,
    enemy_preban: list[str] | None = None,
    *,
    valid_heroes: set[str] | None = None,
    first_pick_team: str = "My Team",
    top_k: int = MAX_BANNABLE_ENEMY_PICKS,
    synergy_lookup: dict[tuple[str, str], SynergyEntry] | None = None,
    response_lookup: dict[tuple[str, str, str], ResponseEntry] | None = None,
    response_evidence: set[str] | None = None,
) -> dict[str, object]:
    candidates, filtered_out = build_ban_candidates(
        ally_picks=ally_picks,
        enemy_picks=enemy_picks,
        first_pick_team=first_pick_team,
        ally_preban=ally_preban,
        enemy_preban=enemy_preban,
        valid_heroes=valid_heroes,
    )

    enemy_context = [
        pick
        for pick in enemy_picks
        if is_valid_hero(str(pick.get("hero") or ""), valid_heroes)
    ]

    recommendations: list[dict[str, object]] = []
    for pick in candidates:
        hero = str(pick["hero"])
        order = int(pick["order"])
        position_bucket = str(pick.get("position_bucket") or get_position_bucket(order))

        synergy_score, synergy_matches = enemy_synergy_core_score(
            hero,
            enemy_context,
            synergy_lookup=synergy_lookup,
        )
        (
            lack_score,
            response_matches,
            team_coverage,
            response_buckets_used,
            response_debug,
        ) = ally_lack_response_score(
            hero,
            ally_picks,
            first_pick_team=first_pick_team,
            position_bucket=position_bucket,
            response_lookup=response_lookup,
            response_evidence=response_evidence,
        )
        threat_score = enemy_position_threat_score(order)
        ban_score = min(
            max(
                WEIGHT_SYNERGY * synergy_score
                + WEIGHT_LACK_RESPONSE * lack_score
                + WEIGHT_POSITION_THREAT * threat_score,
                0.0,
            ),
            1.0,
        )

        recommendations.append(
            {
                "hero": hero,
                "order": order,
                "position_bucket": position_bucket,
                "ban_score": round(ban_score, 6),
                "enemy_synergy_core_score": round(synergy_score, 6),
                "ally_lack_response_score": round(lack_score, 6),
                "enemy_position_threat_score": round(threat_score, 6),
                "reasons": _build_reasons(
                    synergy_score=synergy_score,
                    lack_score=lack_score,
                    threat_score=threat_score,
                    synergy_matches=synergy_matches,
                    response_matches=response_matches,
                    sole_counter_is_bannable=bool(response_debug["sole_counter_is_bannable"]),
                    order=order,
                ),
                "debug": {
                    "synergy_matches": synergy_matches,
                    "response_matches": response_matches,
                    "response_buckets_used": response_buckets_used,
                    "response_sources": response_debug["response_sources"],
                    "sole_counter_is_bannable": response_debug["sole_counter_is_bannable"],
                    "effective_team_response_coverage": response_debug["effective_team_response_coverage"],
                    "original_team_response_coverage": response_debug["original_team_response_coverage"],
                    "team_response_coverage": round(team_coverage, 6),
                    "filtered_out": filtered_out,
                },
            }
        )

    recommendations.sort(key=_sort_key)
    limited = recommendations[: max(top_k, 0)]

    top_rates = _score_to_display_rates(limited)

    return {
        "phase": "ban",
        "handled_by": HANDLED_BY,
        "top_10_heroes": [item["hero"] for item in limited],
        "top_10_rates": top_rates,
        "recommendations": limited,
    }


def recommend_final_bans_from_lists(
    user_team_picks: list[str],
    enemy_team_picks: list[str],
    first_pick_team: str,
    *,
    ally_preban: list[str] | None = None,
    enemy_preban: list[str] | None = None,
    valid_heroes: set[str] | None = None,
    top_k: int = MAX_BANNABLE_ENEMY_PICKS,
) -> dict[str, object]:
    ally_picks = derive_ordered_picks(user_team_picks, first_pick_team=first_pick_team, side="ally")
    enemy_picks = derive_ordered_picks(enemy_team_picks, first_pick_team=first_pick_team, side="enemy")
    return recommend_final_bans(
        ally_picks,
        enemy_picks,
        ally_preban=ally_preban,
        enemy_preban=enemy_preban,
        valid_heroes=valid_heroes,
        first_pick_team=first_pick_team,
        top_k=top_k,
    )
