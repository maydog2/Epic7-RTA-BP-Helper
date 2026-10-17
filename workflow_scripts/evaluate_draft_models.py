"""Offline evaluation for Transformer vs popularity baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "True")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(PROJECT_ROOT)

import numpy as np
import tensorflow as tf

from transformer_draft_data import (
    DATA_DIR,
    DraftEncoders,
    POSITION_BUCKET_ORDER,
    TRANSFORMER_ENCODERS_PATH,
    TRANSFORMER_EVAL_REPORT_PATH_WARFARE,
    TRANSFORMER_TRAIN_NPZ,
    TRANSFORMER_VAL_NPZ,
    TRANSFORMER_WEIGHTS_PATH_WARFARE,
    WARFARE_RULE_ORDER,
    compute_warfare_rule_priors,
    load_npz,
    mask_and_normalize_probs,
    validate_npz_arrays,
)
from transformer_draft_model import arrays_to_model_inputs, build_transformer_draft_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate draft models offline.")
    parser.add_argument("--val-data", type=Path, default=TRANSFORMER_VAL_NPZ)
    parser.add_argument("--train-data", type=Path, default=TRANSFORMER_TRAIN_NPZ)
    parser.add_argument("--transformer-encoders", type=Path, default=TRANSFORMER_ENCODERS_PATH)
    parser.add_argument("--transformer-model", type=Path, default=TRANSFORMER_WEIGHTS_PATH_WARFARE)
    parser.add_argument("--report-out", type=Path, default=TRANSFORMER_EVAL_REPORT_PATH_WARFARE)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    return parser.parse_args()


def masked_topk(probs: np.ndarray, candidate_mask: np.ndarray, k: int) -> np.ndarray:
    masked = probs.copy()
    masked[candidate_mask <= 0] = -np.inf
    return np.argsort(masked)[::-1][:k]


def compute_metrics(
    y_true: np.ndarray,
    prob_matrix: np.ndarray,
    candidate_masks: np.ndarray,
    *,
    draft_steps: np.ndarray | None = None,
    first_pick_sides: np.ndarray | None = None,
    position_bucket_ids: np.ndarray | None = None,
    id_to_position_bucket: list[str] | None = None,
    warfare_rule_ids: np.ndarray | None = None,
    id_to_warfare_rule: list[str] | None = None,
) -> dict:
    n = len(y_true)
    top1 = top3 = top5 = top10 = 0
    nll = 0.0
    valid_recs = 0

    by_step: dict[str, dict[str, float]] = {}
    by_fps: dict[str, dict[str, float]] = {}
    by_position_bucket: dict[str, dict[str, float | int | str | None]] = {
        bucket: {"count": 0, "top1": 0, "top3": 0, "top5": 0}
        for bucket in POSITION_BUCKET_ORDER
    }
    if id_to_position_bucket:
        by_position_bucket["1"]["handled_by"] = "first_pick_stats"
    by_warfare_rule: dict[str, dict[str, float | int]] = {
        rule: {"count": 0, "top1": 0, "top3": 0, "top5": 0}
        for rule in WARFARE_RULE_ORDER
    }

    for i in range(n):
        probs = prob_matrix[i]
        mask = candidate_masks[i]
        target = int(y_true[i])
        top_indices = masked_topk(probs, mask, 10)

        if target in top_indices[:1]:
            top1 += 1
        if target in top_indices[:3]:
            top3 += 1
        if target in top_indices[:5]:
            top5 += 1
        if target in top_indices[:10]:
            top10 += 1

        p = max(float(probs[target]), 1e-12)
        nll += -np.log(p)
        if int(top_indices[0]) not in set(np.where(mask <= 0)[0]):
            valid_recs += 1

        if draft_steps is not None:
            step = int(draft_steps[i])
            bucket = "early" if step <= 3 else "middle" if step <= 6 else "late"
            stats = by_step.setdefault(bucket, {"count": 0, "top1": 0, "top3": 0, "top5": 0})
            stats["count"] += 1
            if target in top_indices[:1]:
                stats["top1"] += 1
            if target in top_indices[:3]:
                stats["top3"] += 1
            if target in top_indices[:5]:
                stats["top5"] += 1

        if first_pick_sides is not None:
            fps_key = "ally_first" if int(first_pick_sides[i]) == 0 else "enemy_first"
            stats = by_fps.setdefault(fps_key, {"count": 0, "top1": 0, "top3": 0, "top5": 0})
            stats["count"] += 1
            if target in top_indices[:1]:
                stats["top1"] += 1
            if target in top_indices[:3]:
                stats["top3"] += 1
            if target in top_indices[:5]:
                stats["top5"] += 1

        if position_bucket_ids is not None and id_to_position_bucket is not None:
            bucket_id = int(position_bucket_ids[i])
            bucket_name = (
                id_to_position_bucket[bucket_id]
                if 0 <= bucket_id < len(id_to_position_bucket)
                else "unknown"
            )
            if bucket_name in by_position_bucket:
                stats = by_position_bucket[bucket_name]
                stats["count"] = int(stats["count"]) + 1
                if target in top_indices[:1]:
                    stats["top1"] = int(stats["top1"]) + 1
                if target in top_indices[:3]:
                    stats["top3"] = int(stats["top3"]) + 1
                if target in top_indices[:5]:
                    stats["top5"] = int(stats["top5"]) + 1

        if warfare_rule_ids is not None and id_to_warfare_rule is not None:
            rule_id = int(warfare_rule_ids[i])
            if 1 <= rule_id < len(id_to_warfare_rule):
                rule_name = id_to_warfare_rule[rule_id]
                if rule_name in by_warfare_rule:
                    stats = by_warfare_rule[rule_name]
                    stats["count"] = int(stats["count"]) + 1
                    if target in top_indices[:1]:
                        stats["top1"] = int(stats["top1"]) + 1
                    if target in top_indices[:3]:
                        stats["top3"] = int(stats["top3"]) + 1
                    if target in top_indices[:5]:
                        stats["top5"] = int(stats["top5"]) + 1

    result = {
        "samples": n,
        "top1": top1 / n,
        "top3": top3 / n,
        "top5": top5 / n,
        "top10": top10 / n,
        "nll": nll / n,
        "candidate_valid_rate": valid_recs / n,
    }

    for bucket, stats in by_step.items():
        count = stats["count"]
        result[f"by_step_{bucket}"] = {
            "samples": count,
            "top1": stats["top1"] / count,
            "top3": stats["top3"] / count,
            "top5": stats["top5"] / count,
        }

    for fps_key, stats in by_fps.items():
        count = stats["count"]
        result[f"by_first_pick_{fps_key}"] = {
            "samples": count,
            "top1": stats["top1"] / count,
            "top3": stats["top3"] / count,
            "top5": stats["top5"] / count,
        }

    for bucket_name, stats in by_position_bucket.items():
        count = int(stats["count"])
        bucket_result = {
            "samples": count,
            "top1": stats["top1"] / count if count else None,
            "top3": stats["top3"] / count if count else None,
            "top5": stats["top5"] / count if count else None,
        }
        if stats.get("handled_by"):
            bucket_result["handled_by"] = stats["handled_by"]
        result[f"by_position_bucket_{bucket_name}"] = bucket_result

    for rule_name, stats in by_warfare_rule.items():
        count = int(stats["count"])
        result[f"by_warfare_rule_{rule_name}"] = {
            "samples": count,
            "top1": stats["top1"] / count if count else None,
            "top3": stats["top3"] / count if count else None,
            "top5": stats["top5"] / count if count else None,
        }

    return result


def build_popularity_baseline(train_arrays: dict[str, np.ndarray], num_heroes: int) -> dict[int, np.ndarray]:
    counts_by_step: dict[int, np.ndarray] = {}
    steps = train_arrays["draft_step"]
    targets = train_arrays["target_next_hero"]
    for step in np.unique(steps):
        step_mask = steps == step
        counts = np.zeros(num_heroes, dtype=np.float64)
        for hero_id in targets[step_mask]:
            counts[int(hero_id)] += 1.0
        if counts.sum() > 0:
            counts /= counts.sum()
        counts_by_step[int(step)] = counts
    return counts_by_step


def predict_popularity(
    val_arrays: dict[str, np.ndarray],
    counts_by_step: dict[int, np.ndarray],
    num_heroes: int,
) -> np.ndarray:
    n = val_arrays["target_next_hero"].shape[0]
    probs = np.zeros((n, num_heroes), dtype=np.float64)
    for i in range(n):
        step = int(val_arrays["draft_step"][i])
        base = counts_by_step.get(step)
        if base is None:
            base = np.ones(num_heroes, dtype=np.float64) / num_heroes
        probs[i] = base
    return probs


def predict_transformer(model, val_arrays: dict[str, np.ndarray]) -> np.ndarray:
    outputs = model.predict(arrays_to_model_inputs(val_arrays), verbose=0)
    if isinstance(outputs, list):
        return outputs[0]
    return outputs


def predict_transformer_any_blended(
    model,
    val_arrays: dict[str, np.ndarray],
    encoders: DraftEncoders,
    priors: dict[str, float],
) -> np.ndarray:
    n = val_arrays["target_next_hero"].shape[0]
    num_heroes = encoders.num_heroes
    blended = np.zeros((n, num_heroes), dtype=np.float64)
    candidate_masks = val_arrays["candidate_mask"]

    for rule in WARFARE_RULE_ORDER:
        rule_arrays = dict(val_arrays)
        rule_arrays["warfare_rule_id"] = np.full(n, encoders.warfare_rule_id_for_name(rule), dtype=np.int32)
        rule_probs = predict_transformer(model, rule_arrays)
        weight = float(priors.get(rule, 0.0))
        if weight <= 0.0:
            continue
        blended += weight * rule_probs

    weight_sum = sum(float(priors.get(rule, 0.0)) for rule in WARFARE_RULE_ORDER)
    if weight_sum <= 0.0:
        blended = np.zeros((n, num_heroes), dtype=np.float64)
        for rule in WARFARE_RULE_ORDER:
            rule_arrays = dict(val_arrays)
            rule_arrays["warfare_rule_id"] = np.full(n, encoders.warfare_rule_id_for_name(rule), dtype=np.int32)
            blended += predict_transformer(model, rule_arrays)
        blended /= len(WARFARE_RULE_ORDER)
    else:
        blended /= weight_sum

    for i in range(n):
        blended[i] = mask_and_normalize_probs(blended[i], candidate_masks[i])
    return blended


def main() -> None:
    args = parse_args()
    encoders = DraftEncoders.load(args.transformer_encoders)
    encoders._require_position_bucket_encoders()
    encoders._require_warfare_rule_encoders()
    train_arrays = load_npz(args.train_data)
    val_arrays = load_npz(args.val_data)
    validate_npz_arrays(val_arrays, split_name="validation")

    y_true = val_arrays["target_next_hero"]
    candidate_masks = val_arrays["candidate_mask"]
    draft_steps = val_arrays["draft_step"]
    first_pick_sides = val_arrays["first_pick_side"]
    position_bucket_ids = val_arrays["current_position_bucket_id"]
    warfare_rule_ids = val_arrays["warfare_rule_id"]
    id_to_position_bucket = encoders.id_to_position_bucket
    id_to_warfare_rule = encoders.id_to_warfare_rule
    priors = encoders.warfare_rule_priors or compute_warfare_rule_priors(
        {rule: int((train_arrays["warfare_rule_id"] == encoders.warfare_rule_id_for_name(rule)).sum())
         for rule in WARFARE_RULE_ORDER}
    )

    popularity_counts = build_popularity_baseline(train_arrays, encoders.num_heroes)
    popularity_probs = predict_popularity(val_arrays, popularity_counts, encoders.num_heroes)

    transformer = build_transformer_draft_model(
        num_heroes=encoders.num_heroes,
        num_roles=encoders.num_roles,
        num_elements=encoders.num_elements,
        num_position_buckets=encoders.num_position_buckets,
        num_warfare_rules=encoders.num_warfare_rules,
        d_model=args.d_model,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
    )
    transformer.load_weights(str(args.transformer_model))
    transformer_probs = predict_transformer(transformer, val_arrays)
    transformer_any_probs = predict_transformer_any_blended(
        transformer,
        val_arrays,
        encoders,
        priors,
    )

    metric_kwargs = {
        "draft_steps": draft_steps,
        "first_pick_sides": first_pick_sides,
        "position_bucket_ids": position_bucket_ids,
        "id_to_position_bucket": id_to_position_bucket,
        "warfare_rule_ids": warfare_rule_ids,
        "id_to_warfare_rule": id_to_warfare_rule,
    }

    report = {
        "validation_samples": int(y_true.shape[0]),
        "num_heroes": encoders.num_heroes,
        "num_position_buckets": encoders.num_position_buckets,
        "num_warfare_rules": encoders.num_warfare_rules,
        "position_buckets": list(POSITION_BUCKET_ORDER),
        "warfare_rules": list(WARFARE_RULE_ORDER),
        "warfare_rule_priors": priors,
        "models": {
            "popularity_baseline": compute_metrics(
                y_true,
                popularity_probs,
                candidate_masks,
                **metric_kwargs,
            ),
            "transformer": compute_metrics(
                y_true,
                transformer_probs,
                candidate_masks,
                **metric_kwargs,
            ),
            "transformer_any_blended_inference": compute_metrics(
                y_true,
                transformer_any_probs,
                candidate_masks,
                draft_steps=draft_steps,
                first_pick_sides=first_pick_sides,
                position_bucket_ids=position_bucket_ids,
                id_to_position_bucket=id_to_position_bucket,
            ),
        },
    }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with args.report_out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved evaluation report to {args.report_out}")


if __name__ == "__main__":
    main()
