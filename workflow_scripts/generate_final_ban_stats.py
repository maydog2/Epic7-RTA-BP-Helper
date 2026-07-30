"""Build layered final-ban exposure/hit statistics from labeled match history."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.runtime_paths import FINAL_BAN_STATS_PATH, RAW_MATCH_HISTORY_PATH, RUNTIME_DATA_DIR  # noqa: E402
from final_ban_stats_core import build_empty_artifact, build_final_ban_stats_from_matches  # noqa: E402
from match_history_utils import enrich_match_draft, validate_match_record  # noqa: E402


def iter_matches(raw_path: Path):
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
            yield enrich_match_draft(match)


def load_matches(raw_path: Path) -> list[dict]:
    matches: list[dict] = []
    for match in iter_matches(raw_path):
        matches.append(match)
    return matches


def summarize_label_coverage(matches: list[dict]) -> dict[str, int | float]:
    total = len(matches)
    labeled = sum(1 for match in matches if match.get("ally_final_ban_target") and match.get("enemy_final_ban_target"))
    coverage = (labeled / total) if total else 0.0
    return {
        "match_count": total,
        "labeled_match_count": labeled,
        "final_ban_label_coverage": round(coverage, 6),
    }


def refresh_final_ban_stats(*, raw_path: Path = RAW_MATCH_HISTORY_PATH) -> dict[str, object]:
    if not raw_path.exists():
        artifact = build_empty_artifact(skip_reasons={"missing_raw_jsonl": 1})
        RUNTIME_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with FINAL_BAN_STATS_PATH.open("wb") as handle:
            pickle.dump(artifact, handle)
        return {
            "final_ban_stats_path": str(FINAL_BAN_STATS_PATH),
            "artifact": artifact,
            "label_coverage": {"match_count": 0, "labeled_match_count": 0, "final_ban_label_coverage": 0.0},
        }

    matches = load_matches(raw_path)
    coverage = summarize_label_coverage(matches)
    artifact = build_final_ban_stats_from_matches(matches)

    RUNTIME_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with FINAL_BAN_STATS_PATH.open("wb") as handle:
        pickle.dump(artifact, handle)

    return {
        "final_ban_stats_path": str(FINAL_BAN_STATS_PATH),
        "artifact": artifact,
        "label_coverage": coverage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate final-ban historical stats artifact.")
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=RAW_MATCH_HISTORY_PATH,
        help=f"Match history JSONL (default: {RAW_MATCH_HISTORY_PATH})",
    )
    args = parser.parse_args()

    summary = refresh_final_ban_stats(raw_path=args.raw_path)
    artifact = summary["artifact"]
    coverage = summary["label_coverage"]

    print(f"Wrote {summary['final_ban_stats_path']}")
    print(
        "Final-ban label coverage: "
        f"{coverage['labeled_match_count']}/{coverage['match_count']} "
        f"({float(coverage['final_ban_label_coverage']) * 100:.2f}%)"
    )
    print(f"Decision count: {artifact['decision_count']}")
    print(f"Handled by: {artifact['handled_by']}")
    if artifact.get("skip_reasons"):
        print(f"Skip reasons: {artifact['skip_reasons']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
