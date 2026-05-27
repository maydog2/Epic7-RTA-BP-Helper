"""Build precomputed runtime artifacts used by the packaged app."""

from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_paths import FIRST_PICK_RECORDS_PATH, PREBAN_STATS_PATH, RUNTIME_DATA_DIR  # noqa: E402
from first_pick_recommender import FirstPickRecord, load_first_pick_records  # noqa: E402
from preban_recommender import load_preban_counts  # noqa: E402


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


def build_runtime_artifacts() -> None:
    RUNTIME_DATA_DIR.mkdir(parents=True, exist_ok=True)

    by_context, combined = load_preban_counts()
    with PREBAN_STATS_PATH.open("wb") as handle:
        pickle.dump(_serialize_preban_counts(by_context, combined), handle)

    records = load_first_pick_records()
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

    print(f"Wrote {PREBAN_STATS_PATH}")
    print(f"Wrote {FIRST_PICK_RECORDS_PATH} ({len(records)} first-pick records)")


if __name__ == "__main__":
    build_runtime_artifacts()
