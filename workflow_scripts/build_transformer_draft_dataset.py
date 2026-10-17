"""Build Transformer draft dataset with match-level train/validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformer_draft_data import (
    POSITION_BUCKET_ORDER,
    RAW_JSONL_PATH,
    TRANSFORMER_ENCODERS_PATH,
    TRANSFORMER_SPLIT_PATH,
    TRANSFORMER_TRAIN_NPZ,
    TRANSFORMER_VAL_NPZ,
    WARFARE_RULE_ORDER,
    build_datasets,
    build_hero_vocab,
    compute_warfare_rule_priors,
    compute_warfare_rule_sample_counts,
    load_transformer_training_matches,
    save_npz,
    split_match_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Transformer draft NPZ datasets.")
    parser.add_argument("--raw", type=Path, default=RAW_JSONL_PATH)
    parser.add_argument("--train-out", type=Path, default=TRANSFORMER_TRAIN_NPZ)
    parser.add_argument("--val-out", type=Path, default=TRANSFORMER_VAL_NPZ)
    parser.add_argument("--meta-out", type=Path, default=TRANSFORMER_ENCODERS_PATH)
    parser.add_argument("--split-meta-out", type=Path, default=TRANSFORMER_SPLIT_PATH)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matches, match_stats = load_transformer_training_matches(args.raw)
    if not matches:
        raise SystemExit("No valid matches with concrete warfare_rules found.")

    encoders = build_hero_vocab(matches)
    match_ids = [int(m["match_id"]) for m in matches]
    train_ids, val_ids = split_match_ids(match_ids, val_ratio=args.val_ratio, seed=args.seed)

    train_arrays, val_arrays = build_datasets(matches, encoders, train_ids, val_ids)
    if not train_arrays or not val_arrays:
        raise SystemExit("Train or validation dataset is empty.")

    per_rule_train_sample_count = compute_warfare_rule_sample_counts(train_arrays)
    per_rule_val_sample_count = compute_warfare_rule_sample_counts(val_arrays)
    warfare_rule_priors = compute_warfare_rule_priors(per_rule_train_sample_count)
    encoders.warfare_rule_priors = warfare_rule_priors

    save_npz(args.train_out, train_arrays)
    save_npz(args.val_out, val_arrays)
    encoders.save(args.meta_out)

    split_meta = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "raw_match_count": match_stats["raw_match_count"],
        "valid_rule_match_count": match_stats["valid_rule_match_count"],
        "skipped_missing_or_empty_warfare_rules_count": match_stats[
            "skipped_missing_or_empty_warfare_rules_count"
        ],
        "skipped_invalid_warfare_rules_count": match_stats["skipped_invalid_warfare_rules_count"],
        "per_rule_match_count": match_stats["per_rule_match_count"],
        "per_rule_train_sample_count": per_rule_train_sample_count,
        "per_rule_val_sample_count": per_rule_val_sample_count,
        "warfare_rule_priors": warfare_rule_priors,
        "train_match_count": len(train_ids),
        "val_match_count": len(val_ids),
        "train_sample_count": int(train_arrays["target_next_hero"].shape[0]),
        "val_sample_count": int(val_arrays["target_next_hero"].shape[0]),
        "num_heroes": encoders.num_heroes,
        "num_position_buckets": encoders.num_position_buckets,
        "num_warfare_rules": encoders.num_warfare_rules,
        "warfare_rules": list(WARFARE_RULE_ORDER),
        "position_buckets": list(POSITION_BUCKET_ORDER),
        "train_match_ids": sorted(train_ids),
        "val_match_ids": sorted(val_ids),
    }
    args.split_meta_out.parent.mkdir(parents=True, exist_ok=True)
    with args.split_meta_out.open("w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)

    print(f"Raw matches: {match_stats['raw_match_count']}")
    print(
        "Skipped missing/empty warfare_rules: "
        f"{match_stats['skipped_missing_or_empty_warfare_rules_count']}"
    )
    print(f"Skipped invalid warfare_rules: {match_stats['skipped_invalid_warfare_rules_count']}")
    print(f"Valid rule matches: {match_stats['valid_rule_match_count']}")
    print(f"Per-rule match counts: {match_stats['per_rule_match_count']}")
    print(f"Per-rule train sample counts: {per_rule_train_sample_count}")
    print(f"Per-rule val sample counts: {per_rule_val_sample_count}")
    print(f"Train rule priors: {warfare_rule_priors}")
    print(f"Matches kept: {len(matches)} | train matches: {len(train_ids)} | val matches: {len(val_ids)}")
    print(f"Train samples: {split_meta['train_sample_count']} -> {args.train_out}")
    print(f"Val samples: {split_meta['val_sample_count']} -> {args.val_out}")
    print(f"Encoders: {args.meta_out}")
    print(f"Split metadata: {args.split_meta_out}")


if __name__ == "__main__":
    main()
