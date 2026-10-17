"""Validate Android recommendation artifacts and Web backend parity fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = PROJECT_ROOT / "workflow_scripts"
if str(WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.final_ban_recommender import recommend_final_bans_from_lists  # noqa: E402
from backend.first_pick_recommender import recommend_first_pick  # noqa: E402
from backend.preban_recommender import (  # noqa: E402
    parse_first_pick_side,
    recommend_prebans,
    resolve_preban_first_pick_side,
)
from backend.recommender_service import (  # noqa: E402
    ensure_recommender_loaded,
    first_pick_recommendation,
    predict_next_hero_transformer,
    reranker_enabled,
    unavailable_heroes_from_draft,
)
from export_android_recommendation_artifacts import export_recommendation_artifacts  # noqa: E402

DEFAULT_ANDROID_ASSETS = PROJECT_ROOT / "E7_BP_Helper_Android" / "app" / "src" / "main" / "assets"
FIXTURES_PATH = WORKFLOW_DIR / "android_recommendation_parity_fixtures.json"
REPORT_PATH = PROJECT_ROOT / "data" / "transformer" / "android_recommendation_parity_report.json"

REQUIRED_ARTIFACTS = [
    "preban_stats.json",
    "first_pick_records.json",
    "reranker_synergy_stats.json",
    "reranker_response_stats.json",
    "reranker_opening_response_stats.json",
    "final_ban_config.json",
    "final_ban_stats.json",
]


def parse_csv(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def run_preban_case(params: dict[str, str]) -> dict:
    raw_side = (params.get("preban_side") or "").strip().lower()
    if raw_side == "user":
        source = "ally"
    elif raw_side == "enemy":
        source = "enemy"
    else:
        source = "combined"
    first_pick_side = resolve_preban_first_pick_side(
        parse_first_pick_side(params.get("first_pick_team")),
        preban_source=source,
    )
    return recommend_prebans(
        parse_csv(params.get("excluded_heroes")),
        int(params.get("top_k") or 10),
        source=source,
        first_pick_side=first_pick_side,
    )


def run_recommend_case(params: dict[str, str]) -> dict:
    user_picks = parse_csv(params.get("user_picks"))
    enemy_picks = parse_csv(params.get("enemy_picks"))
    ally_preban = parse_csv(params.get("ally_preban"))
    enemy_preban = parse_csv(params.get("enemy_preban"))
    first_pick_team = params.get("first_pick_team") or "My Team"
    warfare_rules = params.get("warfare_rules") or "ANY"

    if len(user_picks) >= 5 and len(enemy_picks) >= 5:
        return recommend_final_bans_from_lists(
            user_picks,
            enemy_picks,
            first_pick_team,
            ally_preban=ally_preban,
            enemy_preban=enemy_preban,
            warfare_rules=warfare_rules,
            top_k=4,
        )

    unavailable = unavailable_heroes_from_draft(user_picks, enemy_picks, ally_preban, enemy_preban)
    if (first_pick_team == "My Team" and not user_picks) or (
        first_pick_team == "Enemy Team" and not enemy_picks
    ):
        return first_pick_recommendation(
            first_pick_team=first_pick_team,
            user_team_picks=user_picks,
            enemy_team_picks=enemy_picks,
            ally_preban=ally_preban,
            enemy_preban=enemy_preban,
            unavailable_heroes=unavailable,
        )

    return predict_next_hero_transformer(
        enemy_picks,
        user_picks,
        first_pick_team,
        ally_preban=ally_preban,
        enemy_preban=enemy_preban,
        warfare_rules=warfare_rules,
    )


def validate_payload(case_id: str, payload: dict) -> list[str]:
    errors: list[str] = []
    phase = payload.get("phase")
    if phase not in {"preban", "pick", "ban"}:
        errors.append(f"{case_id}: missing or invalid phase '{phase}'")

    top_heroes = payload.get("top_10_heroes") or []
    top_rates = payload.get("top_10_rates") or []
    if not isinstance(top_heroes, list) or not isinstance(top_rates, list):
        errors.append(f"{case_id}: top_10 payload is not list-shaped")
        return errors

    if len(top_heroes) != len(top_rates):
        errors.append(
            f"{case_id}: top_10 length mismatch heroes={len(top_heroes)} rates={len(top_rates)}"
        )

    if phase == "preban" and not top_heroes:
        errors.append(f"{case_id}: expected non-empty preban recommendations")
    if phase == "pick" and case_id.startswith("first_pick") and not top_heroes:
        errors.append(f"{case_id}: expected non-empty first-pick recommendations")
    if phase == "ban" and len(top_heroes) > 4:
        errors.append(f"{case_id}: final ban should return at most 4 heroes")

    if phase == "pick" and payload.get("reranker_enabled") and "reranker_applied" not in payload:
        if reranker_enabled() and case_id == "normal_pick_rerank":
            errors.append(f"{case_id}: missing reranker_applied flag")

    return errors


def validate_artifacts(output_dir: Path) -> dict[str, object]:
    data_dir = output_dir / "data"
    missing = [name for name in REQUIRED_ARTIFACTS if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Android recommendation artifacts: {missing}")

    preban_stats = json.loads((data_dir / "preban_stats.json").read_text(encoding="utf-8"))
    first_pick_records = json.loads((data_dir / "first_pick_records.json").read_text(encoding="utf-8"))
    synergy_rows = json.loads((data_dir / "reranker_synergy_stats.json").read_text(encoding="utf-8"))

    if not preban_stats.get("by_context"):
        raise ValueError("preban_stats.json has empty by_context")
    if not first_pick_records:
        raise ValueError("first_pick_records.json is empty")
    if not synergy_rows:
        raise ValueError("reranker_synergy_stats.json is empty")

    return {
        "artifacts_ok": True,
        "preban_contexts": len(preban_stats["by_context"]),
        "first_pick_records": len(first_pick_records),
        "synergy_rows": len(synergy_rows),
    }


def run_parity(*, export: bool, output_dir: Path) -> dict[str, object]:
    if export:
        export_recommendation_artifacts(output_dir=output_dir)

    artifact_summary = validate_artifacts(output_dir)
    ensure_recommender_loaded()
    fixtures = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    case_results: list[dict[str, object]] = []
    errors: list[str] = []

    for fixture in fixtures:
        case_id = fixture["id"]
        endpoint = fixture["endpoint"]
        params = fixture["params"]
        if endpoint == "preban":
            payload = run_preban_case(params)
        else:
            payload = run_recommend_case(params)

        case_errors = validate_payload(case_id, payload)
        errors.extend(case_errors)
        case_results.append(
            {
                "id": case_id,
                "endpoint": endpoint,
                "params": params,
                "phase": payload.get("phase"),
                "handled_by": payload.get("handled_by"),
                "top_10_heroes": payload.get("top_10_heroes"),
                "top_10_rates": payload.get("top_10_rates"),
                "errors": case_errors,
            }
        )

    report = {
        "artifact_summary": artifact_summary,
        "reranker_enabled": reranker_enabled(),
        "cases": case_results,
        "passed": not errors,
        "errors": errors,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Android recommendation parity checks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ANDROID_ASSETS,
        help="Android assets output directory.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Validate existing artifacts without re-exporting JSON files.",
    )
    args = parser.parse_args()

    try:
        report = run_parity(export=not args.skip_export, output_dir=args.output)
    except Exception as exc:  # noqa: BLE001
        print(exc, file=sys.stderr)
        return 1

    print(f"Wrote parity report to {REPORT_PATH}")
    print(f"  passed: {report['passed']}")
    print(f"  cases: {len(report['cases'])}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"  ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
