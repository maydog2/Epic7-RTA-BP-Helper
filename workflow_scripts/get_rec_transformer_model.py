"""Train the compact Transformer draft prediction model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "True")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(PROJECT_ROOT)

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from transformer_draft_data import (
    DraftEncoders,
    TRANSFORMER_ENCODERS_PATH,
    TRANSFORMER_TENSORBOARD_DIR_WARFARE,
    TRANSFORMER_TRAIN_NPZ,
    TRANSFORMER_TRAINING_LOG_PATH_WARFARE,
    TRANSFORMER_TRAINING_REPORT_PATH_WARFARE,
    TRANSFORMER_VAL_NPZ,
    TRANSFORMER_WEIGHTS_PATH_WARFARE,
    WARFARE_RULE_ORDER,
    load_npz,
    validate_npz_arrays,
)
from transformer_draft_model import arrays_to_model_inputs, build_transformer_draft_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer draft model.")
    parser.add_argument("--train-data", type=Path, default=TRANSFORMER_TRAIN_NPZ)
    parser.add_argument("--val-data", type=Path, default=TRANSFORMER_VAL_NPZ)
    parser.add_argument("--encoders", type=Path, default=TRANSFORMER_ENCODERS_PATH)
    parser.add_argument("--weights-out", type=Path, default=TRANSFORMER_WEIGHTS_PATH_WARFARE)
    parser.add_argument("--model-out", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--summary-out", type=Path, default=TRANSFORMER_TRAINING_REPORT_PATH_WARFARE)
    parser.add_argument("--csv-log-out", type=Path, default=TRANSFORMER_TRAINING_LOG_PATH_WARFARE)
    parser.add_argument("--tensorboard-log-dir", type=Path, default=TRANSFORMER_TENSORBOARD_DIR_WARFARE)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument(
        "--checkpoint-monitor",
        type=str,
        default="val_top_3_accuracy",
        help="Metric for ModelCheckpoint and EarlyStopping (default: val_top_3_accuracy).",
    )
    parser.add_argument(
        "--lr-monitor",
        type=str,
        default="val_loss",
        help="Metric for ReduceLROnPlateau (default: val_loss).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    if args.model_out is not None:
        args.weights_out = args.model_out.with_suffix(".weights.h5")
    return args


def configure_tensorflow(*, force_cpu: bool = False) -> str:
    if force_cpu:
        tf.config.set_visible_devices([], "GPU")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {[gpu.name for gpu in gpus]}")
        return "gpu"
    os.environ.pop("TF_XLA_FLAGS", None)
    tf.config.optimizer.set_jit(False)
    print("No GPU detected — training on CPU (XLA disabled).")
    return "cpu"


def validate_targets_available(arrays: dict[str, np.ndarray], *, split_name: str) -> None:
    target_ids = arrays["target_next_hero"].astype(np.int64)
    masks = arrays["candidate_mask"]
    row_indices = np.arange(target_ids.shape[0])
    invalid = masks[row_indices, target_ids] <= 0
    if np.any(invalid):
        examples = row_indices[invalid][:10].tolist()
        raise ValueError(
            f"{split_name} has {int(invalid.sum())} targets masked as unavailable; "
            f"example row indices: {examples}"
        )


def make_dataset(
    x_arrays: list[np.ndarray],
    targets: dict[str, np.ndarray],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((tuple(x_arrays), targets))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(x_arrays[0]), seed=seed, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def metric_mode(metric_name: str) -> str:
    if "loss" in metric_name:
        return "min"
    if "accuracy" in metric_name or metric_name.endswith("_rate"):
        return "max"
    return "min"


def best_epoch_summary(history: tf.keras.callbacks.History, monitor: str) -> dict[str, float | int | None]:
    values = history.history.get(monitor, [])
    if not values:
        return {"best_epoch": None, f"best_{monitor}": None}
    mode = metric_mode(monitor)
    best_index = int(np.argmax(values) if mode == "max" else np.argmin(values))
    result = {
        "best_epoch": best_index + 1,
        f"best_{monitor}": float(values[best_index]),
    }
    for extra_metric in ("val_top_3_accuracy", "val_top_5_accuracy", "val_loss"):
        extra_values = history.history.get(extra_metric)
        if extra_values and 0 <= best_index < len(extra_values):
            result[f"best_{extra_metric}"] = float(extra_values[best_index])
    return result


def write_training_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    device: str,
    model: tf.keras.Model,
    history: tf.keras.callbacks.History,
    monitor: str,
    elapsed_seconds: float,
    train_samples: int,
    val_samples: int,
    num_roles: int,
    num_elements: int,
    num_position_buckets: int,
    position_buckets: list[str],
    num_warfare_rules: int,
    warfare_rules: list[str],
) -> None:
    summary = {
        "device": device,
        "weights_out": str(args.weights_out),
        "train_samples": train_samples,
        "validation_samples": val_samples,
        "parameters": int(model.count_params()),
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "ff_dim": args.ff_dim,
            "num_layers": args.num_layers,
            "num_roles": num_roles,
            "num_elements": num_elements,
            "num_position_buckets": num_position_buckets,
            "position_buckets": position_buckets,
            "num_warfare_rules": num_warfare_rules,
            "warfare_rules": warfare_rules,
            "seed": args.seed,
        },
        "monitor": monitor,
        **best_epoch_summary(history, monitor),
        "final_metrics": {
            key: float(values[-1])
            for key, values in history.history.items()
            if values
        },
        "elapsed_seconds": elapsed_seconds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    args = parse_args()
    device = configure_tensorflow(force_cpu=args.force_cpu)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    encoders = DraftEncoders.load(args.encoders)
    encoders._require_position_bucket_encoders()
    encoders._require_warfare_rule_encoders()
    train_arrays = load_npz(args.train_data)
    val_arrays = load_npz(args.val_data)
    validate_npz_arrays(train_arrays, split_name="train")
    validate_npz_arrays(val_arrays, split_name="validation")
    validate_targets_available(train_arrays, split_name="train")
    validate_targets_available(val_arrays, split_name="validation")

    x_train = arrays_to_model_inputs(train_arrays)
    x_val = arrays_to_model_inputs(val_arrays)
    y_train_hero = train_arrays["target_next_hero"]
    y_val_hero = val_arrays["target_next_hero"]
    train_targets = {"next_hero": y_train_hero}
    val_targets = {"next_hero": y_val_hero}

    model = build_transformer_draft_model(
        num_heroes=int(encoders.num_heroes),
        num_roles=int(encoders.num_roles),
        num_elements=int(encoders.num_elements),
        num_position_buckets=int(encoders.num_position_buckets),
        num_warfare_rules=int(encoders.num_warfare_rules),
        d_model=args.d_model,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
    )
    print(f"Model parameters: {model.count_params():,}")
    tf.keras.utils.set_random_seed(args.seed)

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss={"next_hero": "sparse_categorical_crossentropy"},
        metrics={
            "next_hero": [
                "accuracy",
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy"),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name="top_10_accuracy"),
            ],
        },
    )

    checkpoint_monitor = args.checkpoint_monitor
    lr_monitor = args.lr_monitor
    args.weights_out.parent.mkdir(parents=True, exist_ok=True)
    args.csv_log_out.parent.mkdir(parents=True, exist_ok=True)
    args.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath=str(args.weights_out),
            monitor=checkpoint_monitor,
            mode=metric_mode(checkpoint_monitor),
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor=checkpoint_monitor,
            mode=metric_mode(checkpoint_monitor),
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=lr_monitor,
            mode=metric_mode(lr_monitor),
            factor=0.5,
            patience=max(2, args.patience // 2),
            min_lr=1e-5,
            verbose=1,
        ),
        CSVLogger(str(args.csv_log_out)),
        TensorBoard(log_dir=str(args.tensorboard_log_dir), histogram_freq=0),
    ]

    train_dataset = make_dataset(
        x_train,
        train_targets,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_dataset = make_dataset(
        x_val,
        val_targets,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    start_time = time.time()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed_seconds = time.time() - start_time

    if args.weights_out.exists():
        try:
            model.load_weights(args.weights_out)
        except ValueError:
            print(
                f"Checkpoint at {args.weights_out} is incompatible with the current model; "
                "saving weights from the final epoch instead."
            )
            model.save_weights(args.weights_out)
    else:
        model.save_weights(args.weights_out)
    write_training_summary(
        args.summary_out,
        args=args,
        device=device,
        model=model,
        history=history,
        monitor=checkpoint_monitor,
        elapsed_seconds=elapsed_seconds,
        train_samples=len(y_train_hero),
        val_samples=len(y_val_hero),
        num_roles=encoders.num_roles,
        num_elements=encoders.num_elements,
        num_position_buckets=encoders.num_position_buckets,
        position_buckets=encoders.id_to_position_bucket[1:],  # type: ignore[index]
        num_warfare_rules=encoders.num_warfare_rules,
        warfare_rules=list(WARFARE_RULE_ORDER),
    )
    print(f"Saved best Transformer weights to {args.weights_out}")
    print(f"Saved training summary to {args.summary_out}")


if __name__ == "__main__":
    main()
