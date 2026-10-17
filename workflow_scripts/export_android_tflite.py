"""Export the warfare Transformer pick model to TFLite for Android Phase 2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

os.environ.setdefault("TF_USE_LEGACY_KERAS", "True")

import tensorflow as tf  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "workflow_scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "workflow_scripts"))

from transformer_draft_data import (  # noqa: E402
    DraftEncoders,
    TRANSFORMER_ENCODERS_PATH,
    TRANSFORMER_TRAINING_REPORT_PATH_WARFARE,
    TRANSFORMER_WEIGHTS_PATH_WARFARE,
    build_inference_arrays,
    clone_inference_arrays_with_rule,
)
from transformer_draft_model import arrays_to_model_inputs, build_transformer_draft_model  # noqa: E402

DEFAULT_TFLITE_PATH = PROJECT_ROOT / "data" / "transformer" / "rec_model_position_warfare.tflite"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "transformer" / "android_tflite_export_report.json"

KERAS3_LAYER_WEIGHT_PERMUTATION = (
    16, 17, 18, 19, 20, 21, 22, 24, 25, 42, 43, 38, 39, 44, 45, 40, 41, 26, 27, 0, 1, 2, 3, 28,
    29, 50, 51, 46, 47, 52, 53, 48, 49, 30, 31, 4, 5, 6, 7, 32, 33, 58, 59, 54, 55, 60, 61, 56, 57,
    34, 35, 8, 9, 10, 11, 36, 37, 23, 12, 13, 14, 15,
)

MODEL_INPUT_SPECS = (
    {"name": "draft_hero_ids", "dtype": "int32", "shape": [1, 10]},
    {"name": "draft_role_ids", "dtype": "int32", "shape": [1, 10]},
    {"name": "draft_element_ids", "dtype": "int32", "shape": [1, 10]},
    {"name": "draft_side_ids", "dtype": "int32", "shape": [1, 10]},
    {"name": "draft_order_ids", "dtype": "int32", "shape": [1, 10]},
    {"name": "draft_position_bucket_ids", "dtype": "int32", "shape": [1, 10]},
    {"name": "is_first_pick_slot", "dtype": "int32", "shape": [1, 10]},
    {"name": "ally_preban_ids", "dtype": "int32", "shape": [1, 2]},
    {"name": "ally_preban_role_ids", "dtype": "int32", "shape": [1, 2]},
    {"name": "ally_preban_element_ids", "dtype": "int32", "shape": [1, 2]},
    {"name": "enemy_preban_ids", "dtype": "int32", "shape": [1, 2]},
    {"name": "enemy_preban_role_ids", "dtype": "int32", "shape": [1, 2]},
    {"name": "enemy_preban_element_ids", "dtype": "int32", "shape": [1, 2]},
    {"name": "draft_filled_mask", "dtype": "float32", "shape": [1, 10]},
    {"name": "candidate_mask", "dtype": "float32", "shape": [1, None]},
    {"name": "current_position_bucket_id", "dtype": "int32", "shape": [1, 1]},
    {"name": "current_order", "dtype": "int32", "shape": [1, 1]},
    {"name": "current_side_id", "dtype": "int32", "shape": [1, 1]},
    {"name": "warfare_rule_id", "dtype": "int32", "shape": [1, 1]},
)


def build_transformer_model_kwargs(encoders: DraftEncoders) -> dict[str, int]:
    kwargs = {
        "num_heroes": encoders.num_heroes,
        "num_roles": encoders.num_roles,
        "num_elements": encoders.num_elements,
        "num_position_buckets": encoders.num_position_buckets,
        "num_warfare_rules": encoders.num_warfare_rules,
    }
    if TRANSFORMER_TRAINING_REPORT_PATH_WARFARE.is_file():
        with TRANSFORMER_TRAINING_REPORT_PATH_WARFARE.open(encoding="utf-8") as handle:
            config = json.load(handle).get("config", {})
        for key in (
            "d_model",
            "num_heads",
            "ff_dim",
            "num_layers",
            "num_roles",
            "num_elements",
            "num_position_buckets",
            "num_warfare_rules",
        ):
            if key in config:
                kwargs[key] = int(config[key])
    return kwargs


def _read_keras3_layer_weight_arrays(weights_path: Path) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    with h5py.File(weights_path, "r") as weights_file:

        def collect(name: str, obj: object) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            if not name.startswith("layers"):
                return
            if "/vars/" not in name.replace("\\", "/"):
                return
            arrays.append(np.asarray(obj))

        weights_file.visititems(collect)
    return arrays


def assign_keras3_layer_weights(model: tf.keras.Model, weights_path: Path) -> None:
    arrays = _read_keras3_layer_weight_arrays(weights_path)
    expected = len(KERAS3_LAYER_WEIGHT_PERMUTATION)
    if len(arrays) != expected:
        raise RuntimeError(
            f"Expected {expected} Keras 3 layer weight arrays in {weights_path}, found {len(arrays)}"
        )

    trainable_weights = model.trainable_weights
    if len(trainable_weights) != expected:
        raise RuntimeError(
            f"Expected {expected} trainable transformer weights, found {len(trainable_weights)}"
        )

    for model_idx, h5_idx in enumerate(KERAS3_LAYER_WEIGHT_PERMUTATION):
        variable = trainable_weights[model_idx]
        value = arrays[h5_idx]
        if tuple(variable.shape.as_list()) != value.shape:
            raise RuntimeError(
                f"Weight shape mismatch for {variable.name}: model={tuple(variable.shape.as_list())}, "
                f"h5={value.shape}"
            )
        variable.assign(value)


def load_transformer_weights(model: tf.keras.Model, weights_path: Path) -> str:
    errors: list[str] = []
    for by_name in (True, False):
        try:
            model.load_weights(str(weights_path), by_name=by_name)
            return f"keras_load_weights(by_name={by_name})"
        except (ValueError, KeyError, OSError) as exc:
            errors.append(f"by_name={by_name}: {exc}")

    try:
        assign_keras3_layer_weights(model, weights_path)
        return "keras3_layer_assignment"
    except (OSError, RuntimeError) as exc:
        errors.append(f"keras3_assignment: {exc}")
        raise RuntimeError(
            f"Failed to load transformer weights from {weights_path}. "
            f"Attempts: {'; '.join(errors)}"
        ) from exc


def validate_artifact_paths() -> None:
    missing = [
        str(path)
        for path in (
            TRANSFORMER_WEIGHTS_PATH_WARFARE,
            TRANSFORMER_ENCODERS_PATH,
            TRANSFORMER_TRAINING_REPORT_PATH_WARFARE,
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing transformer artifacts:\n  " + "\n  ".join(missing))


def build_loaded_model() -> tuple[tf.keras.Model, DraftEncoders, dict[str, int], str]:
    validate_artifact_paths()
    encoders = DraftEncoders.load(TRANSFORMER_ENCODERS_PATH)
    model_kwargs = build_transformer_model_kwargs(encoders)
    model = build_transformer_draft_model(**model_kwargs)
    load_method = load_transformer_weights(model, TRANSFORMER_WEIGHTS_PATH_WARFARE)
    return model, encoders, model_kwargs, load_method


def sample_inference_inputs(encoders: DraftEncoders) -> list[np.ndarray]:
    arrays = build_inference_arrays(
        encoders,
        user_picks=[],
        enemy_picks=[],
        first_pick_team="My Team",
        ally_preban=[],
        enemy_preban=[],
        warfare_rules="Defense",
    )
    rule_arrays = clone_inference_arrays_with_rule(arrays, encoders, "Defense")
    return arrays_to_model_inputs(rule_arrays)


def normalize_tensor_name(name: str) -> str:
    cleaned = name.split(":")[0]
    if cleaned.startswith("serving_default_"):
        cleaned = cleaned.removeprefix("serving_default_")
    return cleaned


def keras_input_map(model: tf.keras.Model, sample_inputs: list[np.ndarray]) -> dict[str, np.ndarray]:
    if len(model.inputs) != len(sample_inputs):
        raise RuntimeError(
            f"Model has {len(model.inputs)} inputs, sample batch has {len(sample_inputs)}"
        )
    return {
        normalize_tensor_name(inp.name): value
        for inp, value in zip(model.inputs, sample_inputs, strict=True)
    }


def set_interpreter_inputs(
    interpreter: tf.lite.Interpreter,
    input_map: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    for detail in interpreter.get_input_details():
        name = normalize_tensor_name(str(detail["name"]))
        if name not in input_map:
            raise RuntimeError(f"TFLite input {name!r} has no matching Keras sample tensor")
        value = input_map[name]
        interpreter.set_tensor(detail["index"], value.astype(detail["dtype"]))
        applied.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(detail["dtype"]),
            }
        )
    return applied


def convert_model_to_tflite(
    model: tf.keras.Model,
    *,
    sample_inputs: list[np.ndarray],
    use_select_tf_ops: bool,
) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if use_select_tf_ops:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False
    return converter.convert()


def run_interpreter_smoke_test(
    tflite_bytes: bytes,
    input_map: dict[str, np.ndarray],
    *,
    use_select_tf_ops: bool,
) -> dict[str, Any]:
    interpreter = tf.lite.Interpreter(
        model_content=tflite_bytes,
        experimental_preserve_all_tensors=False,
    )
    if use_select_tf_ops:
        interpreter._experimental_disable_delegate_clustering = True
    interpreter.allocate_tensors()

    applied_inputs = set_interpreter_inputs(interpreter, input_map)
    output_details = interpreter.get_output_details()

    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    if output.ndim != 2 or output.shape[0] != 1:
        raise RuntimeError(f"Unexpected TFLite output shape: {output.shape}")

    probs = output[0]
    if not np.isfinite(probs).all():
        raise RuntimeError("TFLite smoke test produced non-finite probabilities")
    if probs.sum() <= 0.0:
        raise RuntimeError("TFLite smoke test produced zero-sum probabilities")

    return {
        "input_count": len(applied_inputs),
        "input_order": [item["name"] for item in applied_inputs],
        "output_shape": list(output.shape),
        "output_sum": float(probs.sum()),
        "output_max": float(probs.max()),
        "top_index": int(np.argmax(probs)),
    }


def compare_keras_and_tflite(
    model: tf.keras.Model,
    tflite_bytes: bytes,
    sample_inputs: list[np.ndarray],
    *,
    use_select_tf_ops: bool,
) -> dict[str, Any]:
    input_map = keras_input_map(model, sample_inputs)
    keras_probs = model.predict(sample_inputs, verbose=0)[0]
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    if use_select_tf_ops:
        interpreter._experimental_disable_delegate_clustering = True
    interpreter.allocate_tensors()
    set_interpreter_inputs(interpreter, input_map)
    interpreter.invoke()
    tflite_probs = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0]

    abs_diff = np.abs(keras_probs - tflite_probs)
    return {
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "keras_top_index": int(np.argmax(keras_probs)),
        "tflite_top_index": int(np.argmax(tflite_probs)),
        "top_index_match": int(np.argmax(keras_probs)) == int(np.argmax(tflite_probs)),
    }


def build_export_report(
    *,
    output_path: Path,
    encoders: DraftEncoders,
    model_kwargs: dict[str, int],
    load_method: str,
    conversion_mode: str,
    smoke_test: dict[str, Any],
    parity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "exported_at": datetime.now(UTC).isoformat(),
        "status": "ok",
        "artifacts": {
            "weights": str(TRANSFORMER_WEIGHTS_PATH_WARFARE),
            "encoders": str(TRANSFORMER_ENCODERS_PATH),
            "training_report": str(TRANSFORMER_TRAINING_REPORT_PATH_WARFARE),
            "tflite": str(output_path),
        },
        "model": {
            "num_heroes": encoders.num_heroes,
            "load_method": load_method,
            **model_kwargs,
        },
        "conversion": {
            "mode": conversion_mode,
            "requires_select_tf_ops": conversion_mode == "select_tf_ops",
        },
        "inputs": [
            {
                **spec,
                "shape": [
                    encoders.num_heroes if dim is None else dim
                    for dim in spec["shape"]
                ],
            }
            for spec in MODEL_INPUT_SPECS
        ],
        "outputs": [
            {
                "name": "next_hero",
                "dtype": "float32",
                "shape": [1, encoders.num_heroes],
            }
        ],
        "smoke_test": smoke_test,
        "export_parity": parity,
        "notes": [
            "Phase 2 export only. Android /api/recommend is not implemented here.",
            "If requires_select_tf_ops is true, Android will need the Flex/Select TF Ops delegate.",
        ],
    }


def export_tflite(
    *,
    output_path: Path,
    report_path: Path,
    force_select_tf_ops: bool,
) -> dict[str, Any]:
    model, encoders, model_kwargs, load_method = build_loaded_model()
    sample_inputs = sample_inference_inputs(encoders)

    conversion_attempts: list[dict[str, str]] = []
    tflite_bytes: bytes | None = None
    conversion_mode = ""

    modes = ["select_tf_ops"] if force_select_tf_ops else ["default", "select_tf_ops"]
    last_error: Exception | None = None
    for mode in modes:
        use_select_tf_ops = mode == "select_tf_ops"
        try:
            tflite_bytes = convert_model_to_tflite(
                model,
                sample_inputs=sample_inputs,
                use_select_tf_ops=use_select_tf_ops,
            )
            conversion_mode = mode
            break
        except Exception as exc:  # noqa: BLE001 - collect conversion blockers for report
            conversion_attempts.append({"mode": mode, "error": str(exc)})
            last_error = exc

    if tflite_bytes is None:
        raise RuntimeError(
            "TFLite conversion failed for all attempted modes: "
            + "; ".join(f"{item['mode']}: {item['error']}" for item in conversion_attempts)
        ) from last_error

    use_select_tf_ops = conversion_mode == "select_tf_ops"
    input_map = keras_input_map(model, sample_inputs)
    smoke_test = run_interpreter_smoke_test(
        tflite_bytes,
        input_map,
        use_select_tf_ops=use_select_tf_ops,
    )
    parity = compare_keras_and_tflite(
        model,
        tflite_bytes,
        sample_inputs,
        use_select_tf_ops=use_select_tf_ops,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_bytes)
    report = build_export_report(
        output_path=output_path,
        encoders=encoders,
        model_kwargs=model_kwargs,
        load_method=load_method,
        conversion_mode=conversion_mode,
        smoke_test=smoke_test,
        parity=parity,
    )
    report["conversion"]["attempts"] = conversion_attempts
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Android Phase 2 TFLite model.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TFLITE_PATH,
        help="Output .tflite path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="JSON export report path.",
    )
    parser.add_argument(
        "--force-select-tf-ops",
        action="store_true",
        help="Skip default conversion and export with SELECT_TF_OPS only.",
    )
    args = parser.parse_args()

    try:
        report = export_tflite(
            output_path=args.output,
            report_path=args.report,
            force_select_tf_ops=args.force_select_tf_ops,
        )
    except Exception as exc:  # noqa: BLE001 - surface export blockers to CLI
        failure_report = {
            "exported_at": datetime.now(UTC).isoformat(),
            "status": "failed",
            "error": str(exc),
            "artifacts": {
                "weights": str(TRANSFORMER_WEIGHTS_PATH_WARFARE),
                "encoders": str(TRANSFORMER_ENCODERS_PATH),
                "training_report": str(TRANSFORMER_TRAINING_REPORT_PATH_WARFARE),
                "tflite": str(args.output),
            },
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(failure_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"TFLite export failed: {exc}", file=sys.stderr)
        print(f"Wrote failure report to {args.report}", file=sys.stderr)
        return 1

    print(f"Exported TFLite model to {args.output}")
    print(f"  conversion mode: {report['conversion']['mode']}")
    print(f"  num_heroes: {report['model']['num_heroes']}")
    print(f"  file size bytes: {args.output.stat().st_size}")
    print(f"  smoke test top index: {report['smoke_test']['top_index']}")
    print(f"  export parity max abs diff: {report['export_parity']['max_abs_diff']:.6g}")
    print(f"Wrote export report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
