"""Update hero_details.csv with pick/preban/appearance counts from match history."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from hero_details_io import (
    DEFAULT_HERO_DETAILS_PATH,
    load_hero_detail_rows,
    save_hero_detail_rows,
)
from match_history_utils import enrich_match_draft, validate_match_record

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "epic7_match_history_raw.jsonl"


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


def compute_appearance_counts(raw_path: Path) -> tuple[Counter[str], Counter[str]]:
    pick_counts: Counter[str] = Counter()
    preban_counts: Counter[str] = Counter()

    for match in iter_valid_matches(raw_path):
        for entry in match.get("draft") or []:
            hero = (entry.get("hero") or "").strip()
            if hero:
                pick_counts[hero] += 1

        for field in ("ally_preban", "enemy_preban"):
            for hero in match.get(field) or []:
                code = (hero or "").strip()
                if code:
                    preban_counts[code] += 1

    return pick_counts, preban_counts


def update_hero_details(
    *,
    raw_path: Path,
    hero_details_path: Path,
) -> dict[str, int]:
    pick_counts, preban_counts = compute_appearance_counts(raw_path)
    rows = load_hero_detail_rows(hero_details_path)

    updated_rows = []
    for row in rows:
        code = row.get("Hero", "")
        appearance_count = pick_counts.get(code, 0) + preban_counts.get(code, 0)
        row["appearance_count"] = str(appearance_count)
        updated_rows.append(row)

    save_hero_detail_rows(updated_rows, hero_details_path)

    heroes_with_appearance = sum(
        1
        for row in updated_rows
        if int(row.get("appearance_count") or 0) > 0
    )
    return {
        "detail_rows": len(updated_rows),
        "heroes_with_appearance": heroes_with_appearance,
        "total_pick_events": sum(pick_counts.values()),
        "total_preban_events": sum(preban_counts.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write appearance_count into data/hero_details.csv. "
            "Prefer workflow_scripts/build_runtime_artifacts.py to refresh "
            "preban, first-pick, and appearance stats together."
        )
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help=f"Match history JSONL (default: {DEFAULT_RAW_PATH})",
    )
    parser.add_argument(
        "--hero-details-path",
        type=Path,
        default=DEFAULT_HERO_DETAILS_PATH,
        help=f"Hero details CSV to update (default: {DEFAULT_HERO_DETAILS_PATH})",
    )
    parser.add_argument(
        "--appearance-only",
        action="store_true",
        help="Only update hero_details.csv (skip preban/first-pick artifacts).",
    )
    args = parser.parse_args()

    if args.appearance_only:
        if not args.raw_path.exists():
            raise SystemExit(f"Match history not found: {args.raw_path}")
        if not args.hero_details_path.exists():
            raise SystemExit(f"Hero details not found: {args.hero_details_path}")

        summary = update_hero_details(
            raw_path=args.raw_path,
            hero_details_path=args.hero_details_path,
        )
        print(
            "Updated hero_details appearance stats: "
            f"{summary['detail_rows']} rows, "
            f"{summary['heroes_with_appearance']} heroes with appearance > 0, "
            f"{summary['total_pick_events']} pick events, "
            f"{summary['total_preban_events']} preban events."
        )
        return

    from build_runtime_artifacts import refresh_match_derived_stats

    try:
        summary = refresh_match_derived_stats(
            raw_path=args.raw_path,
            hero_details_path=args.hero_details_path,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Wrote {summary['preban_stats_path']}")
    print(
        f"Wrote {summary['first_pick_records_path']} "
        f"({summary['first_pick_record_count']} first-pick records)"
    )
    print(
        "Updated hero_details appearance stats: "
        f"{summary['detail_rows']} rows, "
        f"{summary['heroes_with_appearance']} heroes with appearance > 0, "
        f"{summary['total_pick_events']} pick events, "
        f"{summary['total_preban_events']} preban events."
    )


if __name__ == "__main__":
    main()
