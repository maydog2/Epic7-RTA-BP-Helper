"""Refresh match-derived runtime artifacts, reranker stats, and hero appearance from JSONL."""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.first_pick_recommender import FirstPickRecord, load_first_pick_records_from_raw  # noqa: E402
from generate_response_stats import refresh_response_stats  # noqa: E402
from generate_synergy_stats import refresh_synergy_stats  # noqa: E402
from hero_details_io import DEFAULT_HERO_DETAILS_PATH  # noqa: E402
from backend.preban_recommender import load_preban_counts_from_raw  # noqa: E402
from backend.runtime_paths import FIRST_PICK_RECORDS_PATH, PREBAN_STATS_PATH, RUNTIME_DATA_DIR  # noqa: E402
from update_hero_details_appearance_stats import (  # noqa: E402
    DEFAULT_RAW_PATH,
    update_hero_details,
)


GET_HERO_DESCRIPTION_SCRIPT = PROJECT_ROOT / "workflow_scripts" / "get_hero_description.py"
GET_CHARACTER_IDS_SCRIPT = PROJECT_ROOT / "workflow_scripts" / "get_character_ids.py"


def _counter_map(counter: Counter[str]) -> dict[str, int]:
    return dict(counter)


def _serialize_preban_counts(
    by_context: dict[str, dict[str, Counter[str]]],
    combined: dict[str, Counter[str]],
) -> dict[str, object]:
    return {
        "by_context": {
            fps: {side: _counter_map(counter) for side, counter in sides.items()}
            for fps, sides in by_context.items()
        },
        "combined": {side: _counter_map(counter) for side, counter in combined.items()},
    }


def refresh_hero_registry() -> dict[str, object]:
    """Register newly released heroes before rebuilding match-derived stats.

    get_hero_description.py must run before get_character_ids.py: the description
    scraper skips rows that already exist, while get_character_ids.py can insert
    new rows from Stove with empty role/element fields.
    """
    commands = [
        [sys.executable, str(GET_HERO_DESCRIPTION_SCRIPT)],
        [sys.executable, str(GET_CHARACTER_IDS_SCRIPT)],
    ]
    for command in commands:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    return {
        "hero_registration_scripts": [
            str(GET_HERO_DESCRIPTION_SCRIPT),
            str(GET_CHARACTER_IDS_SCRIPT),
        ],
    }


def refresh_match_derived_stats(
    *,
    raw_path: Path = DEFAULT_RAW_PATH,
    hero_details_path: Path = DEFAULT_HERO_DETAILS_PATH,
    skip_hero_registration: bool = False,
    skip_appearance: bool = False,
    skip_reranker: bool = False,
) -> dict[str, object]:
    if not raw_path.exists():
        raise FileNotFoundError(f"Match history not found: {raw_path}")

    summary: dict[str, object] = {
        "raw_path": str(raw_path),
    }

    if not skip_hero_registration:
        summary.update(refresh_hero_registry())

    RUNTIME_DATA_DIR.mkdir(parents=True, exist_ok=True)

    by_context, combined = load_preban_counts_from_raw(raw_path=raw_path)
    with PREBAN_STATS_PATH.open("wb") as handle:
        pickle.dump(_serialize_preban_counts(by_context, combined), handle)

    records = load_first_pick_records_from_raw(raw_path=raw_path)
    with FIRST_PICK_RECORDS_PATH.open("wb") as handle:
        pickle.dump(
            [
                {
                    "first_pick_side": record.first_pick_side,
                    "first_side_preban": record.first_side_preban,
                    "second_side_preban": record.second_side_preban,
                    "order_1_hero": record.order_1_hero,
                    "season": record.season,
                }
                for record in records
            ],
            handle,
        )

    summary.update(
        {
            "preban_stats_path": str(PREBAN_STATS_PATH),
            "first_pick_records_path": str(FIRST_PICK_RECORDS_PATH),
            "first_pick_record_count": len(records),
        }
    )

    if not skip_reranker:
        summary.update(refresh_synergy_stats(raw_path=raw_path))
        summary.update(refresh_response_stats(raw_path=raw_path))

    if skip_appearance:
        return summary

    if not hero_details_path.exists():
        raise FileNotFoundError(f"Hero details not found: {hero_details_path}")

    appearance_summary = update_hero_details(
        raw_path=raw_path,
        hero_details_path=hero_details_path,
    )
    summary["hero_details_path"] = str(hero_details_path)
    summary.update(appearance_summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Register new heroes, then rebuild preban/first-pick runtime artifacts, "
            "reranker CSV stats, and hero_details appearance_count from match history JSONL."
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
        "--skip-hero-registration",
        action="store_true",
        help="Skip get_hero_description.py and get_character_ids.py.",
    )
    parser.add_argument(
        "--skip-appearance",
        action="store_true",
        help="Skip hero_details appearance_count update.",
    )
    parser.add_argument(
        "--skip-reranker",
        action="store_true",
        help="Skip reranker synergy/response CSV regeneration.",
    )
    args = parser.parse_args()

    try:
        summary = refresh_match_derived_stats(
            raw_path=args.raw_path,
            hero_details_path=args.hero_details_path,
            skip_hero_registration=args.skip_hero_registration,
            skip_appearance=args.skip_appearance,
            skip_reranker=args.skip_reranker,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    if args.skip_hero_registration:
        print("Skipped hero registration update.")
    else:
        print("Updated hero registry via get_hero_description.py and get_character_ids.py")

    print(f"Wrote {summary['preban_stats_path']}")
    print(
        f"Wrote {summary['first_pick_records_path']} "
        f"({summary['first_pick_record_count']} first-pick records)"
    )

    if args.skip_reranker:
        print("Skipped reranker synergy/response CSV update.")
    else:
        print(
            f"Wrote {summary['synergy_row_count']} synergy rows to {summary['synergy_stats_path']}"
        )
        print(
            f"Wrote {summary['response_row_count']} response rows to {summary['response_stats_path']}"
        )
        print(
            f"Wrote {summary['opening_response_row_count']} opening rows to "
            f"{summary['opening_response_stats_path']}"
        )

    if args.skip_appearance:
        print("Skipped hero_details appearance_count update.")
        return

    print(
        "Updated hero_details appearance stats: "
        f"{summary['detail_rows']} rows, "
        f"{summary['heroes_with_appearance']} heroes with appearance > 0, "
        f"{summary['total_pick_events']} pick events, "
        f"{summary['total_preban_events']} preban events."
    )


if __name__ == "__main__":
    main()
