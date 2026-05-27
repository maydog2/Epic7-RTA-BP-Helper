"""Historical preban frequency recommendations for RTA draft assistant."""

from __future__ import annotations

import json
import logging
import pickle
from collections import Counter
from pathlib import Path

from runtime_paths import PREBAN_STATS_PATH, RAW_MATCH_HISTORY_PATH

PREBAN_REASON_COMBINED = "High preban frequency in historical RTA matches."
PREBAN_REASON_ALLY = "Often prebanned on your side in historical RTA matches."
PREBAN_REASON_ENEMY = "Often prebanned on the opponent side in historical RTA matches."

# counts[first_pick_side][ban_side] where ban_side is ally | enemy (JSONL field names)
_preban_counts_by_context: dict[str, dict[str, Counter[str]]] | None = None
_preban_counts_combined: dict[str, Counter[str]] | None = None


def parse_top_k(raw_value: str | None, default: int = 10, maximum: int = 50) -> int:
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return min(max(value, 1), maximum)


def parse_first_pick_side(raw_value: str | None) -> str | None:
    """Map UI first_pick_team to JSONL first_pick_side (ally = My Team perspective)."""
    if not raw_value:
        return None
    normalized = raw_value.strip().lower()
    if normalized in ("my team", "ally", "user"):
        return "ally"
    if normalized in ("enemy team", "enemy"):
        return "enemy"
    return None


def resolve_preban_first_pick_side(
    user_first_pick_side: str | None,
    *,
    preban_source: str,
) -> str | None:
    """Draft context keyed by who has first pick in this match.

    Both ally and enemy preban stats are stored under the same first_pick_side
    (from JSONL). preban_source only selects ally_preban vs enemy_preban.
    """
    if user_first_pick_side not in ("ally", "enemy"):
        return None
    return user_first_pick_side


def get_raw_match_history_path() -> Path | None:
    return RAW_MATCH_HISTORY_PATH if RAW_MATCH_HISTORY_PATH.exists() else None


def _counter_from_mapping(mapping: dict[str, int]) -> Counter[str]:
    return Counter(mapping)


def load_preban_counts_from_artifact() -> tuple[dict[str, dict[str, Counter[str]]], dict[str, Counter[str]]] | None:
    if not PREBAN_STATS_PATH.exists():
        return None

    with PREBAN_STATS_PATH.open("rb") as handle:
        payload = pickle.load(handle)

    by_context = {
        fps: {side: _counter_from_mapping(counter) for side, counter in sides.items()}
        for fps, sides in payload["by_context"].items()
    }
    combined = {
        side: _counter_from_mapping(counter) for side, counter in payload["combined"].items()
    }
    return by_context, combined


def _empty_context_counters() -> dict[str, Counter[str]]:
    return {"ally": Counter(), "enemy": Counter()}


def load_preban_counts() -> tuple[dict[str, dict[str, Counter[str]]], dict[str, Counter[str]]]:
    artifact_counts = load_preban_counts_from_artifact()
    if artifact_counts is not None:
        return artifact_counts

    by_context = {"ally": _empty_context_counters(), "enemy": _empty_context_counters()}
    combined = _empty_context_counters()
    raw_path = get_raw_match_history_path()
    if raw_path is None:
        logging.warning("No raw match history JSONL found for preban recommendations")
        return by_context, combined

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

            fps = match.get("first_pick_side")
            if fps not in ("ally", "enemy"):
                continue

            for field, ban_side in (("ally_preban", "ally"), ("enemy_preban", "enemy")):
                prebans = match.get(field) or []
                if not isinstance(prebans, list):
                    continue
                codes = [c for c in prebans if isinstance(c, str) and c]
                by_context[fps][ban_side].update(codes)
                combined[ban_side].update(codes)

    return by_context, combined


def ensure_preban_counts_loaded() -> tuple[dict[str, dict[str, Counter[str]]], dict[str, Counter[str]]]:
    global _preban_counts_by_context, _preban_counts_combined
    if _preban_counts_by_context is None or _preban_counts_combined is None:
        _preban_counts_by_context, _preban_counts_combined = load_preban_counts()
    return _preban_counts_by_context, _preban_counts_combined


def _pick_counter(
    by_context: dict[str, dict[str, Counter[str]]],
    combined: dict[str, Counter[str]],
    *,
    first_pick_side: str | None,
    source: str,
) -> tuple[Counter[str], str, bool]:
    """Return (counter, reason, used_fallback)."""
    if source == "ally":
        reason = PREBAN_REASON_ALLY
        ban_side = "ally"
    elif source == "enemy":
        reason = PREBAN_REASON_ENEMY
        ban_side = "enemy"
    else:
        reason = PREBAN_REASON_COMBINED
        ban_side = None

    if first_pick_side in ("ally", "enemy"):
        ctx = by_context[first_pick_side]
        if ban_side:
            counts = ctx[ban_side]
        else:
            counts = ctx["ally"] + ctx["enemy"]
        if sum(counts.values()) > 0:
            return counts, reason, False

    # Fallback when bucket empty or first_pick_side omitted
    if ban_side:
        counts = combined[ban_side]
    else:
        counts = combined["ally"] + combined["enemy"]
    return counts, reason, True


def recommend_prebans(
    excluded_heroes: list[str],
    top_k: int = 10,
    *,
    source: str = "combined",
    first_pick_side: str | None = None,
) -> dict:
    """Rank prebans; filter by first_pick_side when provided (ally | enemy)."""
    by_context, combined = ensure_preban_counts_loaded()
    counts, reason, used_fallback = _pick_counter(
        by_context, combined, first_pick_side=first_pick_side, source=source
    )

    excluded = set(excluded_heroes)
    total_preban_mentions = sum(counts.values())

    recommendations = []
    if total_preban_mentions > 0:
        for hero_id, count in counts.most_common():
            if hero_id in excluded:
                continue
            share = count / total_preban_mentions
            recommendations.append(
                {
                    "hero_id": hero_id,
                    "normalized_preban_rate": share,
                    "preban_count": count,
                    "reason": reason,
                }
            )
            if len(recommendations) >= top_k:
                break

    return {
        "phase": "preban",
        "preban_source": source,
        "first_pick_side": first_pick_side,
        "preban_stats_fallback": used_fallback,
        "recommendations": recommendations,
        "top_10_heroes": [item["hero_id"] for item in recommendations],
        "top_10_rates": [item["normalized_preban_rate"] * 100.0 for item in recommendations],
    }
