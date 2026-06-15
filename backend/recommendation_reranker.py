"""Conservative inference-side reranker for Transformer Top-10 candidates."""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .preban_recommender import parse_first_pick_side
from .runtime_paths import HERO_DETAILS_PATH, RERANKER_DATA_DIR

SYNERGY_STATS_PATH = RERANKER_DATA_DIR / "hero_synergy_stats.csv"
RESPONSE_STATS_PATH = RERANKER_DATA_DIR / "hero_counter_or_response_stats.csv"
OPENING_RESPONSE_STATS_PATH = RERANKER_DATA_DIR / "enemy_opening_response_stats.csv"

MIN_SYNERGY_COUNT = 3
MIN_RESPONSE_COUNT = 3

LOW_PICK_COUNT_THRESHOLD = int(os.environ.get("LOW_PICK_COUNT_THRESHOLD", "35"))
LOW_PICK_MODEL_CONFIDENCE_THRESHOLD = float(os.environ.get("LOW_PICK_MODEL_CONFIDENCE_THRESHOLD", "0.28"))
LOW_PICK_RESPONSE_THRESHOLD = float(os.environ.get("LOW_PICK_RESPONSE_THRESHOLD", "0.22"))
LOW_PICK_SYNERGY_THRESHOLD = float(os.environ.get("LOW_PICK_SYNERGY_THRESHOLD", "0.22"))
LOW_PICK_GUARD_REASON = "Low-pick conditional hero without enough response/synergy evidence"

BUCKET_WEIGHTS: dict[str, dict[str, float]] = {
    "2_3": {"model": 0.55, "synergy": 0.15, "response": 0.30},
    "4": {"model": 0.65, "synergy": 0.20, "response": 0.15},
    "5_protected": {"model": 0.55, "synergy": 0.35, "response": 0.10},
    "6_protected": {"model": 0.55, "synergy": 0.25, "response": 0.20},
    "7": {"model": 0.60, "synergy": 0.15, "response": 0.25},
    "8_9": {"model": 0.50, "synergy": 0.20, "response": 0.30},
    "10": {"model": 0.50, "synergy": 0.15, "response": 0.35},
}

DEFAULT_BUCKET_WEIGHTS = {"model": 0.60, "synergy": 0.20, "response": 0.20}


@dataclass(frozen=True)
class SynergyEntry:
    score: float
    same_team_count: int


@dataclass(frozen=True)
class ResponseEntry:
    score: float
    response_count: int


@dataclass(frozen=True)
class HeroAppearanceStat:
    appearance_count: int
    appearance_rate: float


_synergy_lookup: dict[tuple[str, str], SynergyEntry] | None = None
_response_lookup: dict[tuple[str, str, str], ResponseEntry] | None = None
_opening_response_lookup: dict[tuple[str, str, str], ResponseEntry] | None = None
_hero_appearance_stats: dict[str, HeroAppearanceStat] | None = None


def low_pick_guard_enabled() -> bool:
    return os.environ.get("RERANKER_LOW_PICK_GUARD", "true").strip().lower() in {"1", "true", "yes"}


def reset_cached_stats() -> None:
    global _synergy_lookup, _response_lookup, _opening_response_lookup, _hero_appearance_stats
    _synergy_lookup = None
    _response_lookup = None
    _opening_response_lookup = None
    _hero_appearance_stats = None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        logging.warning("Reranker artifact not found: %s", path)
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
        entry = SynergyEntry(score=score, same_team_count=same_team_count)
        lookup[(hero_a, hero_b)] = entry
        lookup[(hero_b, hero_a)] = entry

    _synergy_lookup = lookup
    return lookup


def load_response_lookup() -> dict[tuple[str, str, str], ResponseEntry]:
    global _response_lookup
    if _response_lookup is not None:
        return _response_lookup

    lookup: dict[tuple[str, str, str], ResponseEntry] = {}
    for row in _read_csv_rows(RESPONSE_STATS_PATH):
        try:
            response_count = int(float(row["response_count"]))
            score = float(row["normalized_response_score"])
        except (KeyError, TypeError, ValueError):
            continue
        if response_count < MIN_RESPONSE_COUNT:
            continue
        key = (row["position_bucket"], row["candidate_hero"], row["enemy_hero"])
        lookup[key] = ResponseEntry(score=score, response_count=response_count)

    _response_lookup = lookup
    return lookup


def load_hero_appearance_stats() -> dict[str, HeroAppearanceStat]:
    global _hero_appearance_stats
    if _hero_appearance_stats is not None:
        return _hero_appearance_stats

    counts: dict[str, int] = {}
    for row in _read_csv_rows(HERO_DETAILS_PATH):
        hero = row.get("Hero") or row.get("hero")
        if not hero:
            continue
        try:
            counts[hero] = int(float(row.get("appearance_count") or 0))
        except (TypeError, ValueError):
            counts[hero] = 0

    total_appearances = sum(counts.values())
    if total_appearances <= 0:
        stats = {
            hero: HeroAppearanceStat(appearance_count=count, appearance_rate=0.0)
            for hero, count in counts.items()
        }
    else:
        stats = {
            hero: HeroAppearanceStat(
                appearance_count=count,
                appearance_rate=count / total_appearances,
            )
            for hero, count in counts.items()
        }

    _hero_appearance_stats = stats
    return stats


def load_opening_response_lookup() -> dict[tuple[str, str, str], ResponseEntry]:
    global _opening_response_lookup
    if _opening_response_lookup is not None:
        return _opening_response_lookup

    lookup: dict[tuple[str, str, str], ResponseEntry] = {}
    for row in _read_csv_rows(OPENING_RESPONSE_STATS_PATH):
        try:
            response_count = int(float(row["response_count"]))
            score = float(row["normalized_response_score"])
        except (KeyError, TypeError, ValueError):
            continue
        if response_count < MIN_RESPONSE_COUNT:
            continue
        key = (row["first_pick_side"], row["enemy_order_1_hero"], row["candidate_hero"])
        lookup[key] = ResponseEntry(score=score, response_count=response_count)

    _opening_response_lookup = lookup
    return lookup


def bucket_weights(position_bucket: str) -> dict[str, float]:
    return BUCKET_WEIGHTS.get(position_bucket, DEFAULT_BUCKET_WEIGHTS)


def opposing_order_1_hero(
    *,
    first_pick_side: str | None,
    ally_picks: list[str],
    enemy_picks: list[str],
) -> str | None:
    """Return the order-1 hero on the side that opened the draft (opposing side at bucket 2_3)."""
    if first_pick_side == "enemy":
        return enemy_picks[0] if enemy_picks else None
    if first_pick_side == "ally":
        return ally_picks[0] if ally_picks else None
    return None


def derive_first_pick_side(first_pick_team: str | None) -> str | None:
    return parse_first_pick_side(first_pick_team)


def _normalize_model_scores(model_scores: dict[str, float], candidates: list[str]) -> dict[str, float]:
    values = [max(float(model_scores.get(candidate, 0.0)), 0.0) for candidate in candidates]
    total = sum(values)
    if total <= 0.0:
        uniform = 1.0 / len(candidates) if candidates else 0.0
        return {candidate: uniform for candidate in candidates}
    return {candidate: value / total for candidate, value in zip(candidates, values)}


def _score_to_percentages(scores: dict[str, float], ordered_candidates: list[str]) -> list[float]:
    total = sum(max(scores.get(candidate, 0.0), 0.0) for candidate in ordered_candidates)
    if total <= 0.0:
        return [0.0 for _ in ordered_candidates]
    return [(max(scores.get(candidate, 0.0), 0.0) / total) * 100.0 for candidate in ordered_candidates]


def evaluate_low_pick_support(
    *,
    model_score_norm: float,
    ally_synergy_score: float,
    enemy_response_score: float,
) -> dict[str, bool]:
    return {
        "model_confident": model_score_norm >= LOW_PICK_MODEL_CONFIDENCE_THRESHOLD,
        "response_supported": enemy_response_score >= LOW_PICK_RESPONSE_THRESHOLD,
        "synergy_supported": ally_synergy_score >= LOW_PICK_SYNERGY_THRESHOLD,
    }


def reorder_with_low_pick_top3_guard(
    ordered: list[str],
    top3_blocked: dict[str, bool],
) -> list[str]:
    eligible = [candidate for candidate in ordered if not top3_blocked.get(candidate, False)]
    blocked = [candidate for candidate in ordered if top3_blocked.get(candidate, False)]

    top3 = eligible[:3]
    if len(top3) < 3:
        top3.extend(blocked[: 3 - len(top3)])

    used = set(top3)
    rest = [candidate for candidate in ordered if candidate not in used]
    return top3 + rest


def apply_low_pick_top3_guard(
    *,
    ordered: list[str],
    component_scores: dict[str, dict[str, float]],
    debug_records: list[dict[str, object]],
) -> tuple[list[str], list[dict[str, object]]]:
    if not low_pick_guard_enabled():
        return ordered, debug_records

    appearance_stats = load_hero_appearance_stats()
    debug_by_hero = {str(record["hero"]): record for record in debug_records}
    top3_blocked: dict[str, bool] = {}

    for candidate in ordered:
        stat = appearance_stats.get(
            candidate,
            HeroAppearanceStat(appearance_count=0, appearance_rate=0.0),
        )
        scores = component_scores[candidate]
        supporting_evidence = evaluate_low_pick_support(
            model_score_norm=scores["model_score"],
            ally_synergy_score=scores["ally_synergy_score"],
            enemy_response_score=scores["enemy_response_score"],
        )
        is_low_pick = stat.appearance_count < LOW_PICK_COUNT_THRESHOLD
        blocked = is_low_pick and not any(supporting_evidence.values())
        top3_blocked[candidate] = blocked

        record = debug_by_hero[candidate]
        record["appearance_count"] = stat.appearance_count
        record["appearance_rate"] = round(stat.appearance_rate, 6)
        record["low_pick_threshold"] = LOW_PICK_COUNT_THRESHOLD
        record["is_low_pick"] = is_low_pick
        record["top3_blocked"] = blocked
        record["supporting_evidence"] = supporting_evidence
        if blocked:
            record["low_pick_reason"] = LOW_PICK_GUARD_REASON
            record["reason"] = {
                **dict(record.get("reason", {})),
                "low_pick_guard": LOW_PICK_GUARD_REASON,
            }

    final_ordered = reorder_with_low_pick_top3_guard(ordered, top3_blocked)
    return final_ordered, [debug_by_hero[candidate] for candidate in final_ordered]


def _combine_component_scores(scores: list[float], *, position_bucket: str) -> float:
    if not scores:
        return 0.0
    average = sum(scores) / len(scores)
    maximum = max(scores)
    if position_bucket == "10":
        return max(average, maximum)
    return max(average, 0.7 * maximum)


def _lookup_synergy(candidate: str, ally_picks: list[str]) -> tuple[float, list[dict[str, object]]]:
    lookup = load_synergy_lookup()
    matched_scores: list[float] = []
    matches: list[dict[str, object]] = []
    for ally_hero in ally_picks:
        if not ally_hero or ally_hero == "unknown" or ally_hero == candidate:
            continue
        entry = lookup.get((candidate, ally_hero))
        if entry is None:
            continue
        matched_scores.append(entry.score)
        matches.append({"hero": ally_hero, "same_team_count": entry.same_team_count})
    return _combine_component_scores(matched_scores, position_bucket="synergy"), matches


def _lookup_general_response(
    candidate: str,
    enemy_picks: list[str],
    position_bucket: str,
) -> tuple[float, list[dict[str, object]]]:
    lookup = load_response_lookup()
    matched_scores: list[float] = []
    matches: list[dict[str, object]] = []
    for enemy_hero in enemy_picks:
        if not enemy_hero or enemy_hero == "unknown" or enemy_hero == candidate:
            continue
        entry = lookup.get((position_bucket, candidate, enemy_hero))
        if entry is None:
            continue
        matched_scores.append(entry.score)
        matches.append({"hero": enemy_hero, "response_count": entry.response_count})
    return _combine_component_scores(matched_scores, position_bucket=position_bucket), matches


def _lookup_enemy_response_score(
    *,
    candidate: str,
    ally_picks: list[str],
    enemy_picks: list[str],
    position_bucket: str,
    first_pick_side: str | None,
) -> tuple[float, list[dict[str, object]]]:
    if position_bucket == "2_3" and first_pick_side in {"ally", "enemy"}:
        opening_lookup = load_opening_response_lookup()
        enemy_order_1 = opposing_order_1_hero(
            first_pick_side=first_pick_side,
            ally_picks=ally_picks,
            enemy_picks=enemy_picks,
        )
        if enemy_order_1:
            entry = opening_lookup.get((first_pick_side, enemy_order_1, candidate))
            if entry is not None:
                return entry.score, [{"hero": enemy_order_1, "response_count": entry.response_count}]

    return _lookup_general_response(candidate, enemy_picks, position_bucket)


def rerank_candidates(
    *,
    candidates: list[str],
    model_scores: dict[str, float],
    ally_picks: list[str],
    enemy_picks: list[str],
    unavailable_heroes: set[str],
    position_bucket: str,
    first_pick_team: str | None = None,
    top_k: int = 10,
) -> dict[str, object]:
    if position_bucket == "1":
        raise ValueError("Bucket 1 must be handled by first_pick_recommender, not reranker")

    filtered_candidates = [
        candidate
        for candidate in candidates
        if candidate and candidate not in unavailable_heroes and candidate != "unknown"
    ][:top_k]

    if not filtered_candidates:
        return {
            "top_10_heroes": [],
            "top_10_rates": [],
            "debug_recommendations": [],
            "used_reranker": False,
            "position_bucket": position_bucket,
        }

    model_score_norm = _normalize_model_scores(model_scores, filtered_candidates)
    weights = bucket_weights(position_bucket)
    first_pick_side = derive_first_pick_side(first_pick_team)

    debug_records: list[dict[str, object]] = []
    component_scores: dict[str, dict[str, float]] = {}
    has_any_stat_signal = False

    for candidate in filtered_candidates:
        synergy_score, synergy_matches = _lookup_synergy(candidate, ally_picks)
        response_score, response_matches = _lookup_enemy_response_score(
            candidate=candidate,
            ally_picks=ally_picks,
            enemy_picks=enemy_picks,
            position_bucket=position_bucket,
            first_pick_side=first_pick_side,
        )
        if synergy_score > 0.0 or response_score > 0.0:
            has_any_stat_signal = True

        final_score = (
            weights["model"] * model_score_norm[candidate]
            + weights["synergy"] * synergy_score
            + weights["response"] * response_score
        )
        component_scores[candidate] = {
            "model_score": model_score_norm[candidate],
            "ally_synergy_score": synergy_score,
            "enemy_response_score": response_score,
            "final_score": final_score,
        }
        debug_records.append(
            {
                "hero": candidate,
                "model_score": round(model_score_norm[candidate], 6),
                "ally_synergy_score": round(synergy_score, 6),
                "enemy_response_score": round(response_score, 6),
                "final_score": round(final_score, 6),
                "position_bucket": position_bucket,
                "reason": {
                    "synergy_matches": synergy_matches,
                    "response_matches": response_matches,
                    "weights": dict(weights),
                },
            }
        )

    if not has_any_stat_signal:
        ordered = filtered_candidates
        fallback_debug = []
        for candidate in ordered:
            fallback_debug.append(
                {
                    "hero": candidate,
                    "model_score": round(model_score_norm[candidate], 6),
                    "ally_synergy_score": 0.0,
                    "enemy_response_score": 0.0,
                    "final_score": round(model_score_norm[candidate], 6),
                    "position_bucket": position_bucket,
                    "reason": {
                        "synergy_matches": [],
                        "response_matches": [],
                        "weights": dict(weights),
                        "fallback": "no_stat_signal",
                    },
                }
            )
        ordered, ordered_debug = apply_low_pick_top3_guard(
            ordered=ordered,
            component_scores=component_scores,
            debug_records=fallback_debug,
        )
        final_score_map = {candidate: component_scores[candidate]["final_score"] for candidate in ordered}
        rates = _score_to_percentages(final_score_map, ordered)
        return {
            "top_10_heroes": ordered,
            "top_10_rates": [round(rate, 2) for rate in rates],
            "debug_recommendations": ordered_debug,
            "used_reranker": False,
            "position_bucket": position_bucket,
        }

    ordered = sorted(
        filtered_candidates,
        key=lambda candidate: (
            component_scores[candidate]["final_score"],
            component_scores[candidate]["model_score"],
        ),
        reverse=True,
    )
    ordered, ordered_debug = apply_low_pick_top3_guard(
        ordered=ordered,
        component_scores=component_scores,
        debug_records=debug_records,
    )
    final_score_map = {candidate: component_scores[candidate]["final_score"] for candidate in ordered}
    rates = _score_to_percentages(final_score_map, ordered)

    return {
        "top_10_heroes": ordered,
        "top_10_rates": [round(rate, 2) for rate in rates],
        "debug_recommendations": ordered_debug,
        "used_reranker": True,
        "position_bucket": position_bucket,
    }
