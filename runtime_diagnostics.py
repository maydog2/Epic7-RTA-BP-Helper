"""Temporary runtime diagnostics for comparing local vs Render deployments."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from runtime_paths import (
    APP_BASE_DIR,
    BUNDLE_DIR,
    DATA_DIR,
    FIRST_PICK_RECORDS_PATH,
    HERO_DETAILS_PATH,
    PREBAN_STATS_PATH,
    RAW_MATCH_HISTORY_PATH,
    RERANKER_DATA_DIR,
    RUNTIME_DATA_DIR,
    TRANSFORMER_DATA_DIR,
)

MIN_MODEL_BYTES = 1_000_000
MIN_VARIABLES_BYTES = 500
MIN_RUNTIME_PKL_BYTES = 100

RERANKER_CSV_NAMES = (
    "hero_synergy_stats.csv",
    "hero_counter_or_response_stats.csv",
    "enemy_opening_response_stats.csv",
)

OBSOLETE_WARFARE_RULES = {"ANY", "NONE"}


def env_flag(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes"}


def env_value(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value is not None else None


def file_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size": None,
        "is_git_lfs_pointer": False,
        "looks_like_text": False,
    }
    if not path.exists():
        return info

    size = path.stat().st_size
    info["size"] = size
    if path.is_file() and size <= 512:
        try:
            sample = path.read_bytes()[:200]
        except OSError:
            sample = b""
        text = sample.decode("utf-8", errors="ignore")
        info["looks_like_text"] = b"\x00" not in sample
        info["is_git_lfs_pointer"] = text.startswith("version https://git-lfs.github.com/spec/v1")
    return info


def validate_transformer_artifact_paths(
    *,
    model_path: Path,
    variables_path: Path,
) -> None:
    errors: list[str] = []

    model_info = file_info(model_path)
    if not model_info["exists"]:
        errors.append(f"Transformer weights missing: {model_path}")
    elif model_info["is_git_lfs_pointer"]:
        errors.append(f"Transformer weights is a Git LFS pointer, not a real file: {model_path}")
    elif (model_info["size"] or 0) < MIN_MODEL_BYTES:
        errors.append(
            f"Transformer weights too small ({model_info['size']} bytes, expected >= {MIN_MODEL_BYTES}): {model_path}"
        )

    variables_info = file_info(variables_path)
    if not variables_info["exists"]:
        errors.append(f"Transformer encoder missing: {variables_path}")
    elif variables_info["is_git_lfs_pointer"]:
        errors.append(f"Transformer encoder is a Git LFS pointer, not a real file: {variables_path}")
    elif (variables_info["size"] or 0) < MIN_VARIABLES_BYTES:
        errors.append(
            f"Transformer encoder too small ({variables_info['size']} bytes, expected >= {MIN_VARIABLES_BYTES}): {variables_path}"
        )

    if errors:
        raise RuntimeError("Transformer artifact validation failed:\n- " + "\n- ".join(errors))


def validate_loaded_encoders(encoders: Any) -> None:
    encoders._require_position_bucket_encoders()
    encoders._require_warfare_rule_encoders()

    obsolete = sorted(OBSOLETE_WARFARE_RULES.intersection(encoders.warfare_rule_to_id.keys()))
    if obsolete:
        raise RuntimeError(
            "Loaded encoder contains obsolete warfare categories: "
            f"{obsolete}. Expected concrete rules only: Support, Offense, Defense, Resistance."
        )


def reranker_file_infos() -> dict[str, dict[str, Any]]:
    return {name: file_info(RERANKER_DATA_DIR / name) for name in RERANKER_CSV_NAMES}


def runtime_artifact_infos() -> dict[str, dict[str, Any]]:
    infos = {
        "preban_stats_pkl": file_info(PREBAN_STATS_PATH),
        "first_pick_records_pkl": file_info(FIRST_PICK_RECORDS_PATH),
        "raw_match_history_jsonl": file_info(RAW_MATCH_HISTORY_PATH),
    }
    first_pick_info = infos["first_pick_records_pkl"]
    if first_pick_info["exists"] and (first_pick_info["size"] or 0) > 0:
        try:
            import pickle

            with FIRST_PICK_RECORDS_PATH.open("rb") as handle:
                records = pickle.load(handle)
            first_pick_info["record_count"] = len(records) if hasattr(records, "__len__") else None
        except Exception as exc:
            first_pick_info["record_count_error"] = str(exc)
    return infos


def build_runtime_debug_payload(
    *,
    recommender_service_file: str,
    transformer_model_path: Path,
    transformer_variables_path: Path,
    transformer_model_loaded: bool,
    transformer_encoders: Any | None,
    available_heroes: dict[str, Any] | None,
) -> dict[str, Any]:
    model_info = file_info(transformer_model_path)
    variables_info = file_info(transformer_variables_path)
    hero_info = file_info(HERO_DETAILS_PATH)

    payload: dict[str, Any] = {
        "cwd": os.getcwd(),
        "recommender_service_file": recommender_service_file,
        "runtime_paths": {
            "app_base_dir": str(APP_BASE_DIR),
            "bundle_dir": str(BUNDLE_DIR),
            "data_dir": str(DATA_DIR),
            "transformer_data_dir": str(TRANSFORMER_DATA_DIR),
            "runtime_data_dir": str(RUNTIME_DATA_DIR),
            "reranker_data_dir": str(RERANKER_DATA_DIR),
        },
        "artifacts": {
            "model_path": str(transformer_model_path),
            "model_exists": model_info["exists"],
            "model_size": model_info["size"],
            "model_is_git_lfs_pointer": model_info["is_git_lfs_pointer"],
            "variables_path": str(transformer_variables_path),
            "variables_exists": variables_info["exists"],
            "variables_size": variables_info["size"],
            "variables_is_git_lfs_pointer": variables_info["is_git_lfs_pointer"],
            "hero_metadata_path": str(HERO_DETAILS_PATH),
            "hero_metadata_exists": hero_info["exists"],
            "hero_metadata_size": hero_info["size"],
            "runtime_artifacts": runtime_artifact_infos(),
            "reranker_stats": reranker_file_infos(),
        },
        "loaded": {
            "loaded_model_type": "transformer_position_warfare" if transformer_model_loaded else None,
            "transformer_model_loaded": transformer_model_loaded,
            "transformer_encoders_loaded": transformer_encoders is not None,
            "available_hero_count": len(available_heroes or {}),
        },
        "environment": {
            "RECOMMENDER_MODEL": env_value("RECOMMENDER_MODEL"),
            "RECOMMENDER_TRANSFORMER_FALLBACK": env_value("RECOMMENDER_TRANSFORMER_FALLBACK"),
            "RECOMMENDER_RERANKER": env_value("RECOMMENDER_RERANKER"),
            "RECOMMENDER_DEBUG": env_value("RECOMMENDER_DEBUG"),
            "RECOMMENDER_CACHE_SIZE": env_value("RECOMMENDER_CACHE_SIZE"),
            "PYTHON_VERSION": env_value("PYTHON_VERSION"),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
    }

    if transformer_encoders is not None:
        payload["loaded"].update(
            {
                "num_heroes": transformer_encoders.num_heroes,
                "num_position_buckets": transformer_encoders.num_position_buckets,
                "num_warfare_rules": transformer_encoders.num_warfare_rules,
                "warfare_rule_to_id": dict(transformer_encoders.warfare_rule_to_id or {}),
                "position_bucket_to_id": dict(transformer_encoders.position_bucket_to_id or {}),
            }
        )

    return payload


def build_hero_debug_payload(
    hero_code: str,
    *,
    transformer_encoders: Any | None,
    available_heroes: dict[str, Any] | None,
) -> dict[str, Any]:
    in_encoder = False
    hero_id = None
    if transformer_encoders is not None:
        hero_id = transformer_encoders.hero_to_id.get(hero_code)
        in_encoder = hero_id is not None

    return {
        "hero": hero_code,
        "in_encoder": in_encoder,
        "hero_id": hero_id,
        "in_valid_heroes": hero_code in (available_heroes or {}),
        "maps_to_unknown": hero_id == 1 if hero_id is not None else None,
    }


def unrecognized_hero_codes(
    hero_codes: list[str],
    *,
    available_heroes: dict[str, Any],
) -> list[str]:
    return sorted(
        {
            code
            for code in hero_codes
            if code and code not in available_heroes and code not in {"<PAD>", "<UNK>", "unknown"}
        }
    )


def collect_recommend_debug(
    *,
    raw_warfare_rules: str | None,
    normalized_warfare_rules: str,
    user_picks: list[str],
    enemy_picks: list[str],
    ally_preban: list[str],
    enemy_preban: list[str],
    first_pick_team: str,
    from_cache: bool,
    payload: dict[str, object],
    inference_debug: dict[str, Any] | None,
    available_heroes: dict[str, Any],
    reranker_enabled: bool,
    model_path: Path,
    variables_path: Path,
) -> dict[str, Any]:
    request_heroes = [*user_picks, *enemy_picks, *ally_preban, *enemy_preban]
    debug: dict[str, Any] = {
        "normalized_warfare_rules": normalized_warfare_rules,
        "raw_warfare_rules": raw_warfare_rules,
        "request": {
            "user_picks": user_picks,
            "enemy_picks": enemy_picks,
            "ally_preban": ally_preban,
            "enemy_preban": enemy_preban,
            "first_pick_team": first_pick_team,
        },
        "from_cache": from_cache,
        "reranker_enabled": reranker_enabled,
        "reranker_applied": payload.get("reranker_applied"),
        "model_artifact_path": str(model_path),
        "variables_artifact_path": str(variables_path),
        "handled_by": payload.get("handled_by"),
        "fallback_used": payload.get("handled_by") == "first_pick_stats"
        or bool(payload.get("first_pick_fallback_level")),
        "first_pick_fallback_level": payload.get("first_pick_fallback_level"),
        "unrecognized_heroes": unrecognized_hero_codes(request_heroes, available_heroes=available_heroes),
    }
    if inference_debug is not None:
        debug.update(inference_debug)
    return debug
