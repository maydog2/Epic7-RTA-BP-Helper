"""Generate same-team hero synergy statistics for the inference reranker."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from match_history_utils import enrich_match_draft, validate_match_record

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RERANKER_DATA_DIR = PROJECT_ROOT / "data" / "reranker"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "epic7_match_history_raw.jsonl"
DEFAULT_OUTPUT_PATH = RERANKER_DATA_DIR / "hero_synergy_stats.csv"
MIN_SAME_TEAM_COUNT = 3

OUTPUT_COLUMNS = [
    "hero_a",
    "hero_b",
    "same_team_count",
    "hero_a_team_count",
    "hero_b_team_count",
    "normalized_synergy_score",
]


def iter_valid_matches(raw_path: Path):
    with raw_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                match = json.loads(stripped)
            except json.JSONDecodeError as exc:
                print(f"Skipping line {line_number}: invalid JSON ({exc})", file=sys.stderr)
                continue
            errors = validate_match_record(match)
            if errors:
                print(f"Skipping line {line_number}: {errors[0]}", file=sys.stderr)
                continue
            yield enrich_match_draft(match)


def canonical_pair(hero_a: str, hero_b: str) -> tuple[str, str]:
    if hero_a <= hero_b:
        return hero_a, hero_b
    return hero_b, hero_a


def count_team_pairs(team_heroes: list[str], pair_counts: Counter[tuple[str, str]]) -> None:
    unique_heroes = sorted(set(team_heroes))
    for index, hero_a in enumerate(unique_heroes):
        for hero_b in unique_heroes[index + 1 :]:
            pair_counts[canonical_pair(hero_a, hero_b)] += 1


def generate_synergy_stats(raw_path: Path) -> list[dict[str, object]]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    hero_team_counts: Counter[str] = Counter()

    for match in iter_valid_matches(raw_path):
        ally_heroes: list[str] = []
        enemy_heroes: list[str] = []
        for entry in match["draft"]:
            hero = entry["hero"]
            if not hero:
                continue
            if entry["side"] == "ally":
                ally_heroes.append(hero)
            else:
                enemy_heroes.append(hero)

        for hero in set(ally_heroes):
            hero_team_counts[hero] += 1
        for hero in set(enemy_heroes):
            hero_team_counts[hero] += 1

        count_team_pairs(ally_heroes, pair_counts)
        count_team_pairs(enemy_heroes, pair_counts)

    rows: list[dict[str, object]] = []
    for (hero_a, hero_b), same_team_count in sorted(pair_counts.items()):
        if same_team_count < MIN_SAME_TEAM_COUNT:
            continue
        hero_a_team_count = hero_team_counts[hero_a]
        hero_b_team_count = hero_team_counts[hero_b]
        denominator = math.sqrt(hero_a_team_count * hero_b_team_count)
        normalized = same_team_count / denominator if denominator > 0 else 0.0
        rows.append(
            {
                "hero_a": hero_a,
                "hero_b": hero_b,
                "same_team_count": same_team_count,
                "hero_a_team_count": hero_a_team_count,
                "hero_b_team_count": hero_b_team_count,
                "normalized_synergy_score": round(normalized, 6),
            }
        )
    return rows


def write_synergy_stats(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Raw match history JSONL path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path",
    )
    return parser.parse_args()


def refresh_synergy_stats(
    *,
    raw_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, object]:
    rows = generate_synergy_stats(raw_path)
    write_synergy_stats(rows, output_path)
    return {
        "synergy_stats_path": str(output_path),
        "synergy_row_count": len(rows),
    }


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    summary = refresh_synergy_stats(raw_path=args.input, output_path=args.output)
    print(f"Wrote {summary['synergy_row_count']} synergy rows to {summary['synergy_stats_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
