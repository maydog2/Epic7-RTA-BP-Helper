"""Order-1 first-pick recommendations from historical RTA matches."""

from __future__ import annotations

import json
import logging
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from preban_recommender import get_raw_match_history_path
from runtime_paths import FIRST_PICK_RECORDS_PATH

FIRST_PICK_REASON = "Historical order-1 first pick with similar directional preban context."
FALLBACK_LEVEL_WEIGHT = {
    1: 1.0,
    2: 0.50,
    3: 0.25,
    4: 0.05,
}

_first_pick_records: list[FirstPickRecord] | None = None


@dataclass(frozen=True)
class FirstPickRecord:
    first_pick_side: str
    first_side_preban: tuple[str, ...]
    second_side_preban: tuple[str, ...]
    order_1_hero: str
    season: str | None = None


def normalize_preban_list(prebans: list[str] | None, *, max_size: int = 2) -> tuple[str, ...]:
    if not prebans:
        return ()
    return tuple(code for code in prebans if code)[:max_size]


def derive_directional_prebans(
    first_pick_side: str,
    ally_preban: list[str] | None,
    enemy_preban: list[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ally = normalize_preban_list(ally_preban)
    enemy = normalize_preban_list(enemy_preban)
    if first_pick_side == "ally":
        return ally, enemy
    return enemy, ally


def preban_overlap(query: tuple[str, ...], candidate: tuple[str, ...]) -> float:
    if not query and not candidate:
        return 1.0
    if not query or not candidate:
        return 0.0
    overlap = len(set(query) & set(candidate))
    return overlap / max(len(query), len(candidate))


def directional_match_level(
    query_first: tuple[str, ...],
    query_second: tuple[str, ...],
    match_first: tuple[str, ...],
    match_second: tuple[str, ...],
) -> int | None:
    first_overlap = preban_overlap(query_first, match_first)
    second_overlap = preban_overlap(query_second, match_second)

    if query_first == match_first and query_second == match_second:
        return 1
    if query_first == match_first and second_overlap > 0.0 and query_second != match_second:
        return 2
    if first_overlap > 0.0 and second_overlap > 0.0:
        return 3
    return None


def directional_similarity(
    query_first: tuple[str, ...],
    query_second: tuple[str, ...],
    match_first: tuple[str, ...],
    match_second: tuple[str, ...],
) -> float:
    return (
        0.6 * preban_overlap(query_first, match_first)
        + 0.4 * preban_overlap(query_second, match_second)
    )


def load_first_pick_records_from_artifact() -> list[FirstPickRecord] | None:
    if not FIRST_PICK_RECORDS_PATH.exists():
        return None

    with FIRST_PICK_RECORDS_PATH.open("rb") as handle:
        payload = pickle.load(handle)

    return [
        FirstPickRecord(
            first_pick_side=item["first_pick_side"],
            first_side_preban=tuple(item["first_side_preban"]),
            second_side_preban=tuple(item["second_side_preban"]),
            order_1_hero=item["order_1_hero"],
            season=item.get("season"),
        )
        for item in payload
    ]


def load_first_pick_records() -> list[FirstPickRecord]:
    artifact_records = load_first_pick_records_from_artifact()
    if artifact_records is not None:
        return artifact_records

    records: list[FirstPickRecord] = []
    raw_path = get_raw_match_history_path()
    if raw_path is None:
        logging.warning("No raw match history JSONL found for first-pick recommendations")
        return records

    with raw_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                match = json.loads(stripped)
            except json.JSONDecodeError:
                logging.warning("Skipping invalid JSONL line %s in %s", line_number, raw_path)
                continue

            first_pick_side = match.get("first_pick_side")
            if first_pick_side not in ("ally", "enemy"):
                continue

            draft = match.get("draft") or []
            order_one = next((entry for entry in draft if int(entry.get("order", -1)) == 1), None)
            if not order_one:
                continue

            hero = order_one.get("hero")
            if not isinstance(hero, str) or not hero:
                continue

            first_side_preban, second_side_preban = derive_directional_prebans(
                first_pick_side,
                match.get("ally_preban") or [],
                match.get("enemy_preban") or [],
            )
            records.append(
                FirstPickRecord(
                    first_pick_side=first_pick_side,
                    first_side_preban=first_side_preban,
                    second_side_preban=second_side_preban,
                    order_1_hero=hero,
                    season=match.get("season"),
                )
            )

    return records


def ensure_first_pick_records_loaded() -> list[FirstPickRecord]:
    global _first_pick_records
    if _first_pick_records is None:
        _first_pick_records = load_first_pick_records()
    return _first_pick_records


def is_first_pick_recommendation_turn(
    first_pick_team: str,
    user_team_picks: list[str],
    enemy_team_picks: list[str],
) -> bool:
    if first_pick_team == "My Team":
        return len(user_team_picks) == 0
    if first_pick_team == "Enemy Team":
        return len(enemy_team_picks) == 0
    return False


def _format_ranked_recommendations(
    recommendations: list[tuple[str, float]],
    *,
    fallback_level: int,
    filled_through_level: int,
    top_k: int,
) -> dict:
    ranked = sorted(recommendations, key=lambda item: item[1], reverse=True)[:top_k]
    total = sum(weight for _, weight in ranked)
    if total <= 0.0:
        return {
            "phase": "pick",
            "handled_by": "first_pick_stats",
            "first_pick_fallback_level": fallback_level,
            "first_pick_filled_through_level": filled_through_level,
            "top_10_heroes": [],
            "top_10_rates": [],
        }

    top_heroes = [hero for hero, _ in ranked]
    top_rates = [(weight / total) * 100.0 for _, weight in ranked]
    return {
        "phase": "pick",
        "handled_by": "first_pick_stats",
        "first_pick_fallback_level": fallback_level,
        "first_pick_filled_through_level": filled_through_level,
        "reason": FIRST_PICK_REASON,
        "top_10_heroes": top_heroes,
        "top_10_rates": top_rates,
    }


def _append_ranked_counts(
    recommendations: list[tuple[str, float]],
    weighted_counts: Counter[str],
    *,
    excluded_heroes: set[str],
    seen_heroes: set[str],
    fallback_level: int,
) -> None:
    level_weight = FALLBACK_LEVEL_WEIGHT[fallback_level]
    eligible = [
        (hero_code, float(weight))
        for hero_code, weight in weighted_counts.most_common()
        if hero_code not in excluded_heroes and hero_code not in seen_heroes
    ]
    level_total = sum(weight for _, weight in eligible)
    if level_total <= 0.0:
        return

    for hero_code, weight in eligible:
        recommendations.append((hero_code, weight / level_total * level_weight))
        seen_heroes.add(hero_code)


def recommend_first_pick(
    *,
    ally_preban: list[str] | None,
    enemy_preban: list[str] | None,
    first_pick_side: str,
    excluded_heroes: set[str] | None = None,
    top_k: int = 10,
) -> dict:
    if first_pick_side not in ("ally", "enemy"):
        return {
            "phase": "pick",
            "handled_by": "first_pick_stats",
            "first_pick_fallback_level": 4,
            "top_10_heroes": [],
            "top_10_rates": [],
        }

    records = [
        record
        for record in ensure_first_pick_records_loaded()
        if record.first_pick_side == first_pick_side
    ]
    query_first, query_second = derive_directional_prebans(
        first_pick_side,
        ally_preban,
        enemy_preban,
    )
    excluded = excluded_heroes or set()

    ranked_recommendations: list[tuple[str, float]] = []
    seen_heroes: set[str] = set()
    first_used_level: int | None = None
    filled_through_level = 4

    for level in (1, 2, 3):
        weighted_counts: Counter[str] = Counter()
        for record in records:
            match_level = directional_match_level(
                query_first,
                query_second,
                record.first_side_preban,
                record.second_side_preban,
            )
            if match_level != level:
                continue
            weight = 1.0 if level < 3 else directional_similarity(
                query_first,
                query_second,
                record.first_side_preban,
                record.second_side_preban,
            )
            weighted_counts[record.order_1_hero] += weight

        if weighted_counts:
            if first_used_level is None:
                first_used_level = level
            _append_ranked_counts(
                ranked_recommendations,
                weighted_counts,
                excluded_heroes=excluded,
                seen_heroes=seen_heroes,
                fallback_level=level,
            )
            filled_through_level = level

    season_counts: Counter[str] = Counter(record.order_1_hero for record in records)
    if first_used_level is None and season_counts:
        first_used_level = 4
    _append_ranked_counts(
        ranked_recommendations,
        season_counts,
        excluded_heroes=excluded,
        seen_heroes=seen_heroes,
        fallback_level=4,
    )
    return _format_ranked_recommendations(
        ranked_recommendations,
        fallback_level=first_used_level or 4,
        filled_through_level=4,
        top_k=top_k,
    )
