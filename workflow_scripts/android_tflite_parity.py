"""Compare Keras and TFLite Transformer recommendations on fixed draft test cases."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("TF_USE_LEGACY_KERAS", "True")

import tensorflow as tf  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = PROJECT_ROOT / "workflow_scripts"
if str(WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_DIR))

from export_android_tflite import (  # noqa: E402
    DEFAULT_TFLITE_PATH,
    build_loaded_model,
    keras_input_map,
    set_interpreter_inputs,
)
from transformer_draft_data import (  # noqa: E402
    DraftEncoders,
    TRANSFORMER_ENCODERS_PATH,
    WARFARE_RULE_ANY,
    WARFARE_RULE_ORDER,
    blend_concrete_rule_probabilities,
    build_inference_arrays,
    clone_inference_arrays_with_rule,
    compute_warfare_rule_priors,
    mask_and_normalize_probs,
    normalize_warfare_rules_param,
)
from transformer_draft_model import arrays_to_model_inputs  # noqa: E402

DEFAULT_TEST_CASES_PATH = WORKFLOW_DIR / "android_tflite_test_cases.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "transformer" / "android_tflite_parity_report.json"
DEFAULT_EXPORT_REPORT_PATH = PROJECT_ROOT / "data" / "transformer" / "android_tflite_export_report.json"


def unavailable_heroes_from_draft(
    ally_picks: list[str],
    enemy_picks: list[str],
    *,
    ally_preban: list[str] | None = None,
    enemy_preban: list[str] | None = None,
) -> set[str]:
    return {
        code
        for code in [
            *ally_picks,
            *enemy_picks,
            *(ally_preban or []),
            *(enemy_preban or []),
        ]
        if code and code != "unknown"
    }


def normalize_picks(picks: list[str], encoders: DraftEncoders) -> list[str]:
    playable = {
        code
        for code in encoders.hero_to_id
        if code not in {"<PAD>", "<UNK>"}
    }
    return [pick if pick in playable else "unknown" for pick in picks]


def rank_available_recommendations(
    scores: np.ndarray,
    *,
    id_to_hero: list[str],
    hero_to_id: dict[str, int],
    unavailable_heroes: set[str],
    candidate_mask: np.ndarray | None = None,
    k: int = 10,
) -> tuple[list[str], list[float]]:
    masked = scores.astype(np.float64).copy()
    if candidate_mask is not None:
        masked[candidate_mask <= 0] = 0.0

    for hero_code in unavailable_heroes:
        hero_id = hero_to_id.get(hero_code)
        if hero_id is not None and 0 <= hero_id < masked.shape[0]:
            masked[hero_id] = 0.0

    for hero_id, hero_code in enumerate(id_to_hero[: masked.shape[0]]):
        if hero_code in {"", "<PAD>", "<UNK>", "unknown"}:
            masked[hero_id] = 0.0

    total = float(masked.sum())
    if total <= 0.0:
        return [], []

    normalized = masked / total
    top_indices = [idx for idx in np.argsort(normalized)[::-1] if normalized[idx] > 0.0][:k]
    top_heroes = [id_to_hero[idx] for idx in top_indices]
    top_rates = (normalized[top_indices] * 100.0).tolist()
    return top_heroes, top_rates


def run_keras_inference(
    model: tf.keras.Model,
    encoders: DraftEncoders,
    arrays: dict[str, np.ndarray],
    *,
    warfare_rules: str,
) -> np.ndarray:
    normalized_rule = normalize_warfare_rules_param(warfare_rules)
    candidate_mask = arrays["candidate_mask"][0]

    if normalized_rule != WARFARE_RULE_ANY:
        rule_arrays = clone_inference_arrays_with_rule(arrays, encoders, normalized_rule)
        probs = model.predict(arrays_to_model_inputs(rule_arrays), verbose=0)[0]
        return mask_and_normalize_probs(probs, candidate_mask)

    priors = encoders.warfare_rule_priors or compute_warfare_rule_priors(
        {rule: 1 for rule in WARFARE_RULE_ORDER}
    )
    probs_by_rule: dict[str, np.ndarray] = {}
    for rule in WARFARE_RULE_ORDER:
        rule_arrays = clone_inference_arrays_with_rule(arrays, encoders, rule)
        raw_probs = model.predict(arrays_to_model_inputs(rule_arrays), verbose=0)[0]
        probs_by_rule[rule] = mask_and_normalize_probs(raw_probs, candidate_mask)

    blended = blend_concrete_rule_probabilities(probs_by_rule, priors)
    return mask_and_normalize_probs(blended, candidate_mask)


def run_tflite_inference(
    interpreter: tf.lite.Interpreter,
    model: tf.keras.Model,
    encoders: DraftEncoders,
    arrays: dict[str, np.ndarray],
    *,
    warfare_rules: str,
) -> np.ndarray:
    normalized_rule = normalize_warfare_rules_param(warfare_rules)
    candidate_mask = arrays["candidate_mask"][0]

    def predict_once(rule_arrays: dict[str, np.ndarray]) -> np.ndarray:
        sample_inputs = arrays_to_model_inputs(rule_arrays)
        input_map = keras_input_map(model, sample_inputs)
        set_interpreter_inputs(interpreter, input_map)
        interpreter.invoke()
        return interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0]

    if normalized_rule != WARFARE_RULE_ANY:
        rule_arrays = clone_inference_arrays_with_rule(arrays, encoders, normalized_rule)
        probs = predict_once(rule_arrays)
        return mask_and_normalize_probs(probs, candidate_mask)

    priors = encoders.warfare_rule_priors or compute_warfare_rule_priors(
        {rule: 1 for rule in WARFARE_RULE_ORDER}
    )
    probs_by_rule: dict[str, np.ndarray] = {}
    for rule in WARFARE_RULE_ORDER:
        rule_arrays = clone_inference_arrays_with_rule(arrays, encoders, rule)
        raw_probs = predict_once(rule_arrays)
        probs_by_rule[rule] = mask_and_normalize_probs(raw_probs, candidate_mask)

    blended = blend_concrete_rule_probabilities(probs_by_rule, priors)
    return mask_and_normalize_probs(blended, candidate_mask)


def overlap_at_k(left: list[str], right: list[str], k: int) -> dict[str, Any]:
    left_k = left[:k]
    right_k = right[:k]
    left_set = set(left_k)
    right_set = set(right_k)
    shared = left_set & right_set
    union = left_set | right_set
    return {
        "k": k,
        "overlap_count": len(shared),
        "jaccard": (len(shared) / len(union)) if union else 1.0,
        "exact_match": left_k == right_k,
        "shared_heroes": sorted(shared),
    }


def compare_prob_vectors(keras_probs: np.ndarray, tflite_probs: np.ndarray) -> dict[str, float]:
    abs_diff = np.abs(keras_probs - tflite_probs)
    return {
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
    }


def compare_rate_lists(
    keras_heroes: list[str],
    keras_rates: list[float],
    tflite_heroes: list[str],
    tflite_rates: list[float],
) -> dict[str, Any]:
    keras_map = dict(zip(keras_heroes, keras_rates, strict=False))
    tflite_map = dict(zip(tflite_heroes, tflite_rates, strict=False))
    shared = sorted(set(keras_map) & set(tflite_map))
    if not shared:
        return {
            "shared_count": 0,
            "max_rate_abs_diff": None,
            "mean_rate_abs_diff": None,
        }
    diffs = [abs(keras_map[hero] - tflite_map[hero]) for hero in shared]
    return {
        "shared_count": len(shared),
        "max_rate_abs_diff": float(max(diffs)),
        "mean_rate_abs_diff": float(sum(diffs) / len(diffs)),
    }


def load_test_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No test cases found in {path}")
    return cases


def load_tflite_runtime_options(export_report_path: Path) -> dict[str, Any]:
    if not export_report_path.is_file():
        return {"requires_select_tf_ops": False, "conversion_mode": "unknown"}
    report = json.loads(export_report_path.read_text(encoding="utf-8"))
    conversion = report.get("conversion", {})
    return {
        "requires_select_tf_ops": bool(conversion.get("requires_select_tf_ops", False)),
        "conversion_mode": conversion.get("mode", "unknown"),
    }


def evaluate_case(
    case: dict[str, Any],
    *,
    model: tf.keras.Model,
    encoders: DraftEncoders,
    interpreter: tf.lite.Interpreter,
    top_k: int,
) -> dict[str, Any]:
    user_picks = normalize_picks(list(case.get("user_picks") or []), encoders)
    enemy_picks = normalize_picks(list(case.get("enemy_picks") or []), encoders)
    ally_preban = normalize_picks(list(case.get("ally_preban") or []), encoders)
    enemy_preban = normalize_picks(list(case.get("enemy_preban") or []), encoders)
    first_pick_team = str(case.get("first_pick_team") or "My Team")
    warfare_rules = str(case.get("warfare_rules") or WARFARE_RULE_ANY)
    case_top_k = int(case.get("top_k") or top_k)

    arrays = build_inference_arrays(
        encoders,
        user_picks=user_picks,
        enemy_picks=enemy_picks,
        first_pick_team=first_pick_team,
        ally_preban=ally_preban,
        enemy_preban=enemy_preban,
        warfare_rules=warfare_rules,
    )
    unavailable_heroes = unavailable_heroes_from_draft(
        user_picks,
        enemy_picks,
        ally_preban=ally_preban,
        enemy_preban=enemy_preban,
    )
    candidate_mask = arrays["candidate_mask"][0]

    keras_probs = run_keras_inference(
        model,
        encoders,
        arrays,
        warfare_rules=warfare_rules,
    )
    tflite_probs = run_tflite_inference(
        interpreter,
        model,
        encoders,
        arrays,
        warfare_rules=warfare_rules,
    )

    keras_top_heroes, keras_top_rates = rank_available_recommendations(
        keras_probs,
        id_to_hero=encoders.id_to_hero,
        hero_to_id=encoders.hero_to_id,
        unavailable_heroes=unavailable_heroes,
        candidate_mask=candidate_mask,
        k=case_top_k,
    )
    tflite_top_heroes, tflite_top_rates = rank_available_recommendations(
        tflite_probs,
        id_to_hero=encoders.id_to_hero,
        hero_to_id=encoders.hero_to_id,
        unavailable_heroes=unavailable_heroes,
        candidate_mask=candidate_mask,
        k=case_top_k,
    )

    prob_metrics = compare_prob_vectors(keras_probs, tflite_probs)
    rate_metrics = compare_rate_lists(
        keras_top_heroes,
        keras_top_rates,
        tflite_top_heroes,
        tflite_top_rates,
    )

    return {
        "id": case["id"],
        "description": case.get("description", ""),
        "input": {
            "user_picks": user_picks,
            "enemy_picks": enemy_picks,
            "ally_preban": ally_preban,
            "enemy_preban": enemy_preban,
            "first_pick_team": first_pick_team,
            "warfare_rules": warfare_rules,
            "top_k": case_top_k,
        },
        "keras": {
            "top_heroes": keras_top_heroes,
            "top_rates": keras_top_rates,
        },
        "tflite": {
            "top_heroes": tflite_top_heroes,
            "top_rates": tflite_top_rates,
        },
        "probabilities": prob_metrics,
        "ranked_rates": rate_metrics,
        "overlap": {
            "top_1": overlap_at_k(keras_top_heroes, tflite_top_heroes, 1),
            "top_3": overlap_at_k(keras_top_heroes, tflite_top_heroes, 3),
            "top_10": overlap_at_k(keras_top_heroes, tflite_top_heroes, case_top_k),
        },
        "passed": (
            keras_top_heroes[:case_top_k] == tflite_top_heroes[:case_top_k]
            and prob_metrics["max_abs_diff"] <= 1e-4
        ),
    }


def summarize_results(case_results: list[dict[str, Any]], *, default_top_k: int) -> dict[str, Any]:
    total = len(case_results)
    passed = sum(1 for item in case_results if item["passed"])
    top1_matches = sum(1 for item in case_results if item["overlap"]["top_1"]["exact_match"])
    top3_matches = sum(1 for item in case_results if item["overlap"]["top_3"]["exact_match"])
    topk_matches = sum(1 for item in case_results if item["overlap"]["top_10"]["exact_match"])
    return {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "top_1_exact_matches": top1_matches,
        "top_3_exact_matches": top3_matches,
        "top_k_exact_matches": topk_matches,
        "default_top_k": default_top_k,
        "all_passed": passed == total,
    }


def run_parity(
    *,
    test_cases_path: Path,
    tflite_path: Path,
    report_path: Path,
    export_report_path: Path,
    top_k: int,
) -> dict[str, Any]:
    if not tflite_path.is_file():
        raise FileNotFoundError(f"Missing TFLite model: {tflite_path}")

    cases = load_test_cases(test_cases_path)
    model, encoders, model_kwargs, load_method = build_loaded_model()
    runtime_options = load_tflite_runtime_options(export_report_path)

    interpreter = tf.lite.Interpreter(model_content=tflite_path.read_bytes())
    if runtime_options["requires_select_tf_ops"]:
        interpreter._experimental_disable_delegate_clustering = True
    interpreter.allocate_tensors()

    case_results = [
        evaluate_case(
            case,
            model=model,
            encoders=encoders,
            interpreter=interpreter,
            top_k=top_k,
        )
        for case in cases
    ]

    summary = summarize_results(case_results, default_top_k=top_k)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "ok" if summary["all_passed"] else "failed",
        "artifacts": {
            "test_cases": str(test_cases_path),
            "tflite": str(tflite_path),
            "encoders": str(TRANSFORMER_ENCODERS_PATH),
            "export_report": str(export_report_path) if export_report_path.is_file() else None,
        },
        "runtime": runtime_options,
        "model": {
            "load_method": load_method,
            **model_kwargs,
        },
        "summary": summary,
        "cases": case_results,
        "notes": [
            "Baseline uses the existing Keras Transformer inference path without reranker.",
            "Top-k ranking mirrors backend rank_available_recommendations().",
            "A case passes only when ranked heroes match exactly and max_abs_diff <= 1e-4.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Android Phase 2 Keras vs TFLite parity checks.")
    parser.add_argument(
        "--test-cases",
        type=Path,
        default=DEFAULT_TEST_CASES_PATH,
        help="JSON file containing fixed draft test cases.",
    )
    parser.add_argument(
        "--tflite",
        type=Path,
        default=DEFAULT_TFLITE_PATH,
        help="Exported TFLite model path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Parity report output path.",
    )
    parser.add_argument(
        "--export-report",
        type=Path,
        default=DEFAULT_EXPORT_REPORT_PATH,
        help="Optional export report used to detect SELECT_TF_OPS runtime needs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Default top-k ranking size for cases without an explicit top_k field.",
    )
    args = parser.parse_args()

    try:
        report = run_parity(
            test_cases_path=args.test_cases,
            tflite_path=args.tflite,
            report_path=args.report,
            export_report_path=args.export_report,
            top_k=args.top_k,
        )
    except Exception as exc:  # noqa: BLE001 - surface blockers in CLI/report
        failure_report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "status": "error",
            "error": str(exc),
            "artifacts": {
                "test_cases": str(args.test_cases),
                "tflite": str(args.tflite),
                "encoders": str(TRANSFORMER_ENCODERS_PATH),
                "export_report": str(args.export_report) if args.export_report.is_file() else None,
            },
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(failure_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Parity run failed: {exc}", file=sys.stderr)
        print(f"Wrote failure report to {args.report}", file=sys.stderr)
        return 1

    summary = report["summary"]
    print(f"Wrote parity report to {args.report}")
    print(f"  cases: {summary['passed_cases']}/{summary['total_cases']} passed")
    print(f"  top-1 exact matches: {summary['top_1_exact_matches']}/{summary['total_cases']}")
    print(f"  top-3 exact matches: {summary['top_3_exact_matches']}/{summary['total_cases']}")
    if not summary["all_passed"]:
        failed_ids = [item["id"] for item in report["cases"] if not item["passed"]]
        print(f"  failed case ids: {', '.join(failed_ids)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
