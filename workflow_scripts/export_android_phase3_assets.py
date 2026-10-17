"""Export Phase 3A Android assets: TFLite model and encoder metadata."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = PROJECT_ROOT / "workflow_scripts"
if str(WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_DIR))

from transformer_draft_data import (  # noqa: E402
    ALLY_SIDE_ID,
    DRAFT_SLOTS,
    ENEMY_SIDE_ID,
    FIRST_PICK_SLOT_INDICES,
    MAX_PREBAN,
    PAD_HERO_ID,
    SIDE_ORDER_ALLY_FIRST,
    SIDE_ORDER_ENEMY_FIRST,
    TRANSFORMER_ENCODERS_PATH,
    UNK_HERO_ID,
    WARFARE_RULE_ORDER,
    DraftEncoders,
)

DEFAULT_ANDROID_ASSETS = PROJECT_ROOT / "E7_BP_Helper_Android" / "app" / "src" / "main" / "assets"
TFLITE_SOURCE = PROJECT_ROOT / "data" / "transformer" / "rec_model_position_warfare.tflite"
EXPORT_REPORT_PATH = PROJECT_ROOT / "data" / "transformer" / "android_tflite_export_report.json"
PARITY_REPORT_PATH = PROJECT_ROOT / "data" / "transformer" / "android_tflite_parity_report.json"
TEST_CASES_PATH = WORKFLOW_DIR / "android_tflite_test_cases.json"


def load_export_report(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing export report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_encoder_metadata(encoders: DraftEncoders, export_report: dict) -> dict:
    encoders._require_position_bucket_encoders()
    encoders._require_warfare_rule_encoders()
    conversion = export_report.get("conversion", {})
    smoke_test = export_report.get("smoke_test", {})
    return {
        "version": 1,
        "draft_slots": DRAFT_SLOTS,
        "max_preban": MAX_PREBAN,
        "pad_hero_id": PAD_HERO_ID,
        "unk_hero_id": UNK_HERO_ID,
        "ally_side_id": int(ALLY_SIDE_ID),
        "enemy_side_id": int(ENEMY_SIDE_ID),
        "first_pick_slot_indices": list(FIRST_PICK_SLOT_INDICES),
        "side_order_ally_first": [int(x) for x in SIDE_ORDER_ALLY_FIRST.tolist()],
        "side_order_enemy_first": [int(x) for x in SIDE_ORDER_ENEMY_FIRST.tolist()],
        "draft_position_bucket_ids": [int(x) for x in encoders.draft_position_bucket_ids().tolist()],
        "position_bucket_to_id": encoders.position_bucket_to_id,
        "hero_to_id": encoders.hero_to_id,
        "id_to_hero": encoders.id_to_hero,
        "hero_role_ids": [int(x) for x in encoders.hero_role_ids.tolist()],
        "hero_element_ids": [int(x) for x in encoders.hero_element_ids.tolist()],
        "warfare_rule_order": list(WARFARE_RULE_ORDER),
        "warfare_rule_to_id": encoders.warfare_rule_to_id,
        "warfare_rule_priors": encoders.warfare_rule_priors
        or {rule: 1.0 / len(WARFARE_RULE_ORDER) for rule in WARFARE_RULE_ORDER},
        "model": {
            "num_heroes": encoders.num_heroes,
            "num_roles": encoders.num_roles,
            "num_elements": encoders.num_elements,
            "num_position_buckets": encoders.num_position_buckets,
            "num_warfare_rules": encoders.num_warfare_rules,
            "tflite_asset_path": "data/rec_model_position_warfare.tflite",
            "requires_select_tf_ops": bool(conversion.get("requires_select_tf_ops", False)),
            "conversion_mode": conversion.get("mode", "unknown"),
            "inputs": export_report.get("inputs", []),
            "outputs": export_report.get("outputs", []),
            "tflite_input_order": smoke_test.get("input_order", []),
        },
        "ranking": {
            "default_top_k": 10,
            "mask_special_heroes": ["", "<PAD>", "<UNK>", "unknown"],
        },
    }


def build_debug_case(parity_report: dict) -> dict | None:
    cases = parity_report.get("cases") or []
    if not cases:
        return None
    first_case = cases[0]
    return {
        "id": first_case["id"],
        "description": first_case.get("description", ""),
        "input": first_case["input"],
        "expected_top_heroes": first_case["tflite"]["top_heroes"],
        "expected_top_rates": first_case["tflite"]["top_rates"],
    }


def export_phase3_assets(*, output_dir: Path) -> dict[str, str | int]:
    if not TFLITE_SOURCE.is_file():
        raise FileNotFoundError(f"Missing TFLite model: {TFLITE_SOURCE}")
    if not TRANSFORMER_ENCODERS_PATH.is_file():
        raise FileNotFoundError(f"Missing encoder pickle: {TRANSFORMER_ENCODERS_PATH}")
    if not PARITY_REPORT_PATH.is_file():
        raise FileNotFoundError(f"Missing parity report: {PARITY_REPORT_PATH}")
    parity_report = json.loads(PARITY_REPORT_PATH.read_text(encoding="utf-8"))
    if parity_report.get("status") == "error" and not parity_report.get("cases"):
        raise FileNotFoundError(
            "TFLite parity report is in error state. Re-run export_android_tflite.py "
            "and android_tflite_parity.py before exporting Phase 3 assets."
        )

    export_report = load_export_report(EXPORT_REPORT_PATH)
    encoders = DraftEncoders.load(TRANSFORMER_ENCODERS_PATH)

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    tflite_dest = data_dir / "rec_model_position_warfare.tflite"
    shutil.copy2(TFLITE_SOURCE, tflite_dest)

    metadata_path = data_dir / "transformer_encoder_metadata.json"
    metadata_path.write_text(
        json.dumps(build_encoder_metadata(encoders, export_report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    debug_case_path = data_dir / "android_tflite_debug_case.json"
    debug_case = build_debug_case(parity_report)
    if debug_case is not None:
        debug_case_path.write_text(
            json.dumps(debug_case, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    test_cases_dest = data_dir / "android_tflite_test_cases.json"
    shutil.copy2(TEST_CASES_PATH, test_cases_dest)

    return {
        "tflite": str(tflite_dest),
        "metadata": str(metadata_path),
        "debug_case": str(debug_case_path) if debug_case is not None else "",
        "test_cases": str(test_cases_dest),
        "tflite_bytes": tflite_dest.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Android Phase 3A TFLite assets.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ANDROID_ASSETS,
        help="Android assets output directory.",
    )
    args = parser.parse_args()

    try:
        stats = export_phase3_assets(output_dir=args.output)
    except Exception as exc:  # noqa: BLE001
        print(exc, file=sys.stderr)
        return 1

    print(f"Exported Phase 3A assets to {args.output}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
