"""Export Android-friendly JSON recommendation artifacts from runtime pickle/CSV data."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANDROID_ASSETS = PROJECT_ROOT / "E7_BP_Helper_Android" / "app" / "src" / "main" / "assets"
RUNTIME_DATA_DIR = PROJECT_ROOT / "data" / "runtime"
RERANKER_DATA_DIR = PROJECT_ROOT / "data" / "reranker"

PREBAN_STATS_PATH = RUNTIME_DATA_DIR / "preban_stats.pkl"
FIRST_PICK_RECORDS_PATH = RUNTIME_DATA_DIR / "first_pick_records.pkl"
FINAL_BAN_STATS_PATH = RUNTIME_DATA_DIR / "final_ban_stats.pkl"
SYNERGY_STATS_PATH = RERANKER_DATA_DIR / "hero_synergy_stats.csv"
RESPONSE_STATS_PATH = RERANKER_DATA_DIR / "hero_counter_or_response_stats.csv"
OPENING_RESPONSE_STATS_PATH = RERANKER_DATA_DIR / "enemy_opening_response_stats.csv"


def counter_to_dict(counter: dict[str, int] | object) -> dict[str, int]:
    if isinstance(counter, dict):
        return {str(key): int(value) for key, value in counter.items()}
    return {str(key): int(value) for key, value in counter.items()}


def export_preban_stats() -> dict[str, object]:
    if not PREBAN_STATS_PATH.is_file():
        raise FileNotFoundError(f"Missing preban stats artifact: {PREBAN_STATS_PATH}")
    with PREBAN_STATS_PATH.open("rb") as handle:
        payload = pickle.load(handle)

    by_context = {
        fps: {side: counter_to_dict(counter) for side, counter in sides.items()}
        for fps, sides in payload["by_context"].items()
    }
    combined = {
        side: counter_to_dict(counter) for side, counter in payload["combined"].items()
    }
    return {"by_context": by_context, "combined": combined}


def export_first_pick_records() -> list[dict[str, object]]:
    if not FIRST_PICK_RECORDS_PATH.is_file():
        raise FileNotFoundError(f"Missing first-pick records artifact: {FIRST_PICK_RECORDS_PATH}")
    with FIRST_PICK_RECORDS_PATH.open("rb") as handle:
        payload = pickle.load(handle)

    records: list[dict[str, object]] = []
    for item in payload:
        records.append(
            {
                "first_pick_side": item["first_pick_side"],
                "first_side_preban": list(item["first_side_preban"]),
                "second_side_preban": list(item["second_side_preban"]),
                "order_1_hero": item["order_1_hero"],
                "season": item.get("season"),
            }
        )
    return records


def export_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing reranker artifact: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def export_synergy_stats() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in export_csv_rows(SYNERGY_STATS_PATH):
        rows.append(
            {
                "hero_a": row["hero_a"],
                "hero_b": row["hero_b"],
                "same_team_count": int(float(row["same_team_count"])),
                "normalized_synergy_score": float(row["normalized_synergy_score"]),
            }
        )
    return rows


def export_response_stats() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in export_csv_rows(RESPONSE_STATS_PATH):
        rows.append(
            {
                "position_bucket": row["position_bucket"],
                "candidate_hero": row["candidate_hero"],
                "enemy_hero": row["enemy_hero"],
                "response_count": int(float(row["response_count"])),
                "normalized_response_score": float(row["normalized_response_score"]),
            }
        )
    return rows


def export_opening_response_stats() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in export_csv_rows(OPENING_RESPONSE_STATS_PATH):
        rows.append(
            {
                "first_pick_side": row["first_pick_side"],
                "enemy_order_1_hero": row["enemy_order_1_hero"],
                "candidate_hero": row["candidate_hero"],
                "response_count": int(float(row["response_count"])),
                "normalized_response_score": float(row["normalized_response_score"]),
            }
        )
    return rows


def export_final_ban_config(*, artifact: dict[str, object] | None = None) -> dict[str, object]:
    source = artifact
    if source is None and FINAL_BAN_STATS_PATH.is_file():
        with FINAL_BAN_STATS_PATH.open("rb") as handle:
            source = pickle.load(handle)
    hybrid_config = dict((source or {}).get("hybrid_config") or {})
    has_labels = bool((source or {}).get("has_labeled_decisions"))
    handled_by = (source or {}).get("handled_by") if has_labels else "final_ban_stats_v1"
    return {
        "version": 2,
        "max_bannable_enemy_picks": 4,
        "min_synergy_count": 3,
        "min_response_count": 3,
        "weight_synergy": 0.48,
        "weight_lack_response": 0.40,
        "weight_position_threat": 0.12,
        "ban_rate_exponent": 3.0,
        "ban_rate_rank_weights": [1.35, 1.0, 0.78, 0.62],
        "response_bucket_weights": {"7": 0.20, "8_9": 0.40, "10": 0.40},
        "position_bucket_response_weights": {
            "2_3": {"2_3": 1.0},
            "4": {"4": 1.0},
            "7": {"7": 1.0},
            "8_9": {"8_9": 1.0},
            "10": {"10": 1.0},
        },
        "order_threat_scores": {
            "2": 0.45,
            "3": 0.50,
            "4": 0.55,
            "7": 0.62,
            "8": 0.72,
            "9": 0.72,
            "10": 0.78,
        },
        "first_pick_orders": [1, 4, 5, 8, 9],
        "second_pick_orders": [2, 3, 6, 7, 10],
        "prior_strength": hybrid_config.get("prior_strength", 10.0),
        "confidence_strength": hybrid_config.get("confidence_strength", 20.0),
        "max_history_weight": hybrid_config.get("max_history_weight", 0.70),
        "handled_by": handled_by,
    }


def export_final_ban_stats() -> dict[str, object]:
    if not FINAL_BAN_STATS_PATH.is_file():
        return {
            "version": 1,
            "has_labeled_decisions": False,
            "decision_count": 0,
            "levels": {level: {} for level in ("level_1", "level_2", "level_3", "level_4")},
            "totals": {"ban_count": 0, "eligible_count": 0},
        }
    with FINAL_BAN_STATS_PATH.open("rb") as handle:
        artifact = pickle.load(handle)
    levels = artifact.get("levels") or {}
    exported_levels: dict[str, dict[str, dict[str, int]]] = {}
    for level_name in ("level_1", "level_2", "level_3", "level_4"):
        store = levels.get(level_name) or {}
        exported_levels[level_name] = {
            key: {
                "ban_count": int(entry.get("ban_count") or 0),
                "eligible_count": int(entry.get("eligible_count") or 0),
            }
            for key, entry in store.items()
        }
    return {
        "version": artifact.get("version", 1),
        "has_labeled_decisions": bool(artifact.get("has_labeled_decisions")),
        "decision_count": int(artifact.get("decision_count") or 0),
        "levels": exported_levels,
        "totals": {
            "ban_count": int((artifact.get("totals") or {}).get("ban_count") or 0),
            "eligible_count": int((artifact.get("totals") or {}).get("eligible_count") or 0),
        },
    }


def export_recommendation_artifacts(*, output_dir: Path) -> dict[str, str | int]:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    final_ban_stats = export_final_ban_stats()
    artifacts = {
        "preban_stats.json": export_preban_stats(),
        "first_pick_records.json": export_first_pick_records(),
        "reranker_synergy_stats.json": export_synergy_stats(),
        "reranker_response_stats.json": export_response_stats(),
        "reranker_opening_response_stats.json": export_opening_response_stats(),
        "final_ban_stats.json": final_ban_stats,
        "final_ban_config.json": export_final_ban_config(),
    }

    stats: dict[str, str | int] = {}
    for filename, payload in artifacts.items():
        path = data_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        stats[filename] = str(path)
        if isinstance(payload, list):
            stats[f"{filename}_rows"] = len(payload)
        elif filename == "preban_stats.json":
            stats["preban_contexts"] = len(payload["by_context"])

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Android recommendation JSON artifacts.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ANDROID_ASSETS,
        help="Android assets output directory.",
    )
    args = parser.parse_args()

    try:
        stats = export_recommendation_artifacts(output_dir=args.output)
    except Exception as exc:  # noqa: BLE001
        print(exc, file=sys.stderr)
        return 1

    print(f"Exported recommendation artifacts to {args.output / 'data'}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
