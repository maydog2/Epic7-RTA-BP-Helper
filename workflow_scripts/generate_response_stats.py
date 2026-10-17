"""Generate bucket-aware response/counter statistics for the inference reranker."""

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
DEFAULT_RESPONSE_OUTPUT = RERANKER_DATA_DIR / "hero_counter_or_response_stats.csv"
DEFAULT_OPENING_OUTPUT = RERANKER_DATA_DIR / "enemy_opening_response_stats.csv"
MIN_RESPONSE_COUNT = 3

RESPONSE_COLUMNS = [
    "position_bucket",
    "candidate_hero",
    "enemy_hero",
    "response_count",
    "candidate_bucket_count",
    "enemy_seen_count",
    "normalized_response_score",
]

OPENING_COLUMNS = [
    "first_pick_side",
    "enemy_order_1_hero",
    "candidate_hero",
    "response_count",
    "normalized_response_score",
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


def generate_response_stats(raw_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    response_counts: Counter[tuple[str, str, str]] = Counter()
    candidate_bucket_counts: Counter[tuple[str, str]] = Counter()
    enemy_seen_counts: Counter[tuple[str, str]] = Counter()

    opening_response_counts: Counter[tuple[str, str, str]] = Counter()
    opening_candidate_counts: Counter[tuple[str, str]] = Counter()
    opening_enemy_order1_counts: Counter[tuple[str, str]] = Counter()

    for match in iter_valid_matches(raw_path):
        first_pick_side = match["first_pick_side"]
        draft_by_order = {int(entry["order"]): entry for entry in match["draft"]}
        if 1 not in draft_by_order:
            continue

        order_1_hero = draft_by_order[1]["hero"]
        visible_ally: list[str] = []
        visible_enemy: list[str] = []

        for order in sorted(draft_by_order):
            entry = draft_by_order[order]
            hero = entry["hero"]
            side = entry["side"]
            bucket = entry["position_bucket"]

            if order == 1:
                if side == "ally":
                    visible_ally.append(hero)
                else:
                    visible_enemy.append(hero)
                continue

            enemy_visible = visible_enemy if side == "ally" else visible_ally
            for enemy_hero in enemy_visible:
                response_counts[(bucket, hero, enemy_hero)] += 1
                enemy_seen_counts[(bucket, enemy_hero)] += 1
            candidate_bucket_counts[(bucket, hero)] += 1

            if bucket == "2_3":
                opening_response_counts[(first_pick_side, order_1_hero, hero)] += 1
                opening_candidate_counts[(first_pick_side, hero)] += 1
                opening_enemy_order1_counts[(first_pick_side, order_1_hero)] += 1

            if side == "ally":
                visible_ally.append(hero)
            else:
                visible_enemy.append(hero)

    response_rows: list[dict[str, object]] = []
    for (bucket, candidate, enemy), response_count in sorted(response_counts.items()):
        if response_count < MIN_RESPONSE_COUNT:
            continue
        candidate_bucket_count = candidate_bucket_counts[(bucket, candidate)]
        enemy_seen_count = enemy_seen_counts[(bucket, enemy)]
        denominator = math.sqrt(candidate_bucket_count * enemy_seen_count)
        normalized = response_count / denominator if denominator > 0 else 0.0
        response_rows.append(
            {
                "position_bucket": bucket,
                "candidate_hero": candidate,
                "enemy_hero": enemy,
                "response_count": response_count,
                "candidate_bucket_count": candidate_bucket_count,
                "enemy_seen_count": enemy_seen_count,
                "normalized_response_score": round(normalized, 6),
            }
        )

    opening_rows: list[dict[str, object]] = []
    for (first_pick_side, enemy_order_1_hero, candidate), response_count in sorted(
        opening_response_counts.items()
    ):
        if response_count < MIN_RESPONSE_COUNT:
            continue
        candidate_count = opening_candidate_counts[(first_pick_side, candidate)]
        enemy_order1_count = opening_enemy_order1_counts[(first_pick_side, enemy_order_1_hero)]
        denominator = math.sqrt(candidate_count * enemy_order1_count)
        normalized = response_count / denominator if denominator > 0 else 0.0
        opening_rows.append(
            {
                "first_pick_side": first_pick_side,
                "enemy_order_1_hero": enemy_order_1_hero,
                "candidate_hero": candidate,
                "response_count": response_count,
                "normalized_response_score": round(normalized, 6),
            }
        )

    return response_rows, opening_rows


def write_csv(rows: list[dict[str, object]], output_path: Path, columns: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def print_top_pairs(rows: list[dict[str, object]], bucket: str, *, limit: int = 5) -> None:
    filtered = [row for row in rows if row["position_bucket"] == bucket]
    filtered.sort(key=lambda row: row["normalized_response_score"], reverse=True)
    print(f"\nTop {limit} response pairs for bucket {bucket}:")
    for row in filtered[:limit]:
        print(
            f"  {row['candidate_hero']} vs {row['enemy_hero']}: "
            f"score={row['normalized_response_score']}, count={row['response_count']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--response-output", type=Path, default=DEFAULT_RESPONSE_OUTPUT)
    parser.add_argument("--opening-output", type=Path, default=DEFAULT_OPENING_OUTPUT)
    parser.add_argument(
        "--print-top",
        action="store_true",
        help="Print top response pairs for buckets 2_3, 8_9, and 10",
    )
    return parser.parse_args()


def refresh_response_stats(
    *,
    raw_path: Path = DEFAULT_INPUT_PATH,
    response_output: Path = DEFAULT_RESPONSE_OUTPUT,
    opening_output: Path = DEFAULT_OPENING_OUTPUT,
) -> dict[str, object]:
    response_rows, opening_rows = generate_response_stats(raw_path)
    write_csv(response_rows, response_output, RESPONSE_COLUMNS)
    write_csv(opening_rows, opening_output, OPENING_COLUMNS)
    return {
        "response_stats_path": str(response_output),
        "response_row_count": len(response_rows),
        "opening_response_stats_path": str(opening_output),
        "opening_response_row_count": len(opening_rows),
    }


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    summary = refresh_response_stats(
        raw_path=args.input,
        response_output=args.response_output,
        opening_output=args.opening_output,
    )
    print(
        f"Wrote {summary['response_row_count']} response rows to {summary['response_stats_path']}"
    )
    print(
        f"Wrote {summary['opening_response_row_count']} opening rows to "
        f"{summary['opening_response_stats_path']}"
    )

    if args.print_top:
        for bucket in ("2_3", "8_9", "10"):
            print_top_pairs(response_rows, bucket)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
