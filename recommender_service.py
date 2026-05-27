import csv
import json
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "True")

import numpy as np
import tensorflow as tf
from flask import Flask, abort, jsonify, request, send_from_directory

from first_pick_recommender import is_first_pick_recommendation_turn, recommend_first_pick
from final_ban_recommender import recommend_final_bans_from_lists
from preban_recommender import parse_first_pick_side, parse_top_k, recommend_prebans, resolve_preban_first_pick_side
from recommendation_reranker import rerank_candidates
from runtime_diagnostics import (
    build_hero_debug_payload,
    build_runtime_debug_payload,
    collect_recommend_debug,
    validate_loaded_encoders,
    validate_transformer_artifact_paths,
)
from runtime_paths import (
    APP_BASE_DIR,
    ELEMENT_ICON_DIR,
    FRONTEND_DIST_DIR,
    HERO_DETAILS_PATH,
    PORTRAIT_DIR,
    ROLE_ICON_DIR,
    WORKFLOW_DIR,
)

if str(WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_DIR))

from transformer_draft_data import (  # noqa: E402
    DraftEncoders,
    TRANSFORMER_ENCODERS_PATH,
    TRANSFORMER_TRAINING_REPORT_PATH_WARFARE,
    TRANSFORMER_WEIGHTS_PATH_WARFARE,
    WARFARE_RULE_ANY,
    WARFARE_RULE_ORDER,
    blend_concrete_rule_probabilities,
    build_inference_arrays,
    clone_inference_arrays_with_rule,
    compute_warfare_rule_priors,
    mask_and_normalize_probs,
    normalize_warfare_rules_param,
)
from transformer_draft_model import arrays_to_model_inputs, build_transformer_draft_model  # noqa: E402


TRANSFORMER_MODEL_PATH = TRANSFORMER_WEIGHTS_PATH_WARFARE
TRANSFORMER_VARIABLES_PATH = TRANSFORMER_ENCODERS_PATH

# CeciliaBot / legacy scraper codes that differ from hero_details.csv
HERO_DETAIL_ALIASES = {
    "c0002": "c1005",  # Mercedes
}

app = Flask(__name__)
logging.basicConfig(
    filename=APP_BASE_DIR / "recommender_service.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

transformer_model = None
transformer_encoders = None
available_heroes = {}
_recommendation_cache: OrderedDict[tuple, dict[str, object]] = OrderedDict()


def recommendation_cache_max_size() -> int:
    raw = os.environ.get("RECOMMENDER_CACHE_SIZE", "512")
    try:
        return max(int(raw), 1)
    except ValueError:
        return 512


def recommendation_cache_key(
    *,
    user_picks: list[str],
    enemy_picks: list[str],
    ally_preban: list[str],
    enemy_preban: list[str],
    first_pick_team: str,
    warfare_rules: str,
) -> tuple:
    """Cache key uses normalized pick lists; preban codes are sorted for stable matching."""
    return (
        tuple(user_picks),
        tuple(enemy_picks),
        tuple(sorted(ally_preban)),
        tuple(sorted(enemy_preban)),
        first_pick_team,
        warfare_rules,
        reranker_enabled(),
        debug_recommendations_enabled(),
    )


def get_cached_recommendation(key: tuple) -> dict[str, object] | None:
    cached = _recommendation_cache.get(key)
    if cached is None:
        return None
    _recommendation_cache.move_to_end(key)
    return cached


def store_cached_recommendation(key: tuple, payload: dict[str, object]) -> None:
    _recommendation_cache[key] = payload
    _recommendation_cache.move_to_end(key)
    while len(_recommendation_cache) > recommendation_cache_max_size():
        _recommendation_cache.popitem(last=False)


def clear_recommendation_cache() -> None:
    _recommendation_cache.clear()


def reranker_enabled() -> bool:
    return os.environ.get("RECOMMENDER_RERANKER", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def debug_recommendations_enabled() -> bool:
    return os.environ.get("RECOMMENDER_DEBUG", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def maybe_rerank_transformer_recommendations(
    *,
    top_10_heroes: list[str],
    top_10_rates: list[float],
    hero_probs: np.ndarray,
    arrays: dict[str, np.ndarray],
    user_team_picks: list[str],
    enemy_team_picks: list[str],
    unavailable_heroes: set[str],
    first_pick_team: str,
) -> dict[str, object]:
    if not reranker_enabled() or not top_10_heroes or transformer_encoders is None:
        return {
            "top_10_heroes": top_10_heroes,
            "top_10_rates": top_10_rates,
        }

    bucket_id = int(arrays["current_position_bucket_id"][0])
    position_bucket = transformer_encoders.id_to_position_bucket[bucket_id]
    if position_bucket == "1":
        return {
            "top_10_heroes": top_10_heroes,
            "top_10_rates": top_10_rates,
        }

    model_scores = {
        hero: float(hero_probs[transformer_encoders.hero_to_id.get(hero, 1)])
        for hero in top_10_heroes
    }
    rerank_result = rerank_candidates(
        candidates=top_10_heroes,
        model_scores=model_scores,
        ally_picks=user_team_picks,
        enemy_picks=enemy_team_picks,
        unavailable_heroes=unavailable_heroes,
        position_bucket=position_bucket,
        first_pick_team=first_pick_team,
        top_k=10,
    )
    payload: dict[str, object] = {
        "top_10_heroes": rerank_result["top_10_heroes"],
        "top_10_rates": rerank_result["top_10_rates"],
        "reranker_enabled": True,
        "reranker_applied": rerank_result["used_reranker"],
    }
    if debug_recommendations_enabled():
        payload["debug_recommendations"] = rerank_result["debug_recommendations"]
    return payload


def get_allowed_origins() -> list[str]:
    raw_origins = os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    allowed_origins = get_allowed_origins()
    if origin in allowed_origins or "*" in allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = origin if origin else "*"
        response.headers["Vary"] = "Origin"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def parse_picks(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [pick for pick in raw_value.split(",") if pick]


def resolve_hero_row(code: str, rows_by_code: dict[str, dict[str, str]]) -> dict[str, str]:
    if code in rows_by_code:
        return rows_by_code[code]
    alias = HERO_DETAIL_ALIASES.get(code)
    if alias and alias in rows_by_code:
        return rows_by_code[alias]
    return {}


def hero_metadata_for_code(
    code: str,
    row: dict[str, str],
    rows_by_code: dict[str, dict[str, str]],
) -> tuple[str, str, int]:
    role = (row.get("Role") or "").strip()
    element = (row.get("Element") or "").strip()
    appearance_count = parse_non_negative_int(row.get("appearance_count"))

    alias_code = HERO_DETAIL_ALIASES.get(code)
    if alias_code and alias_code in rows_by_code:
        alias_row = rows_by_code[alias_code]
        if not role:
            role = (alias_row.get("Role") or "").strip()
        if not element:
            element = (alias_row.get("Element") or "").strip()
        if appearance_count <= 0:
            appearance_count = parse_non_negative_int(alias_row.get("appearance_count"))

    fallback = resolve_hero_row(code, rows_by_code)
    if not role:
        role = (fallback.get("Role") or "").strip()
    if not element:
        element = (fallback.get("Element") or "").strip()
    if appearance_count <= 0:
        appearance_count = parse_non_negative_int(fallback.get("appearance_count"))

    return role, element, appearance_count


def parse_non_negative_int(raw_value: str | None) -> int:
    if raw_value is None:
        return 0
    try:
        return max(int(float(raw_value)), 0)
    except (TypeError, ValueError):
        return 0


def load_hero_options() -> list[dict[str, object]]:
    if not HERO_DETAILS_PATH.exists():
        return []

    rows_by_code: dict[str, dict[str, str]] = {}
    with HERO_DETAILS_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = (row.get("Hero") or "").strip()
            if code:
                rows_by_code[code] = row

    heroes: list[dict[str, object]] = []
    for code in sorted(rows_by_code):
        row = rows_by_code[code]
        english_name = (row.get("name") or "").strip()
        chinese_name = (row.get("name_zh") or "").strip()
        display_name = english_name or chinese_name
        if not display_name:
            continue
        role, element, appearance_count = hero_metadata_for_code(code, row, rows_by_code)
        heroes.append(
            {
                "code": code,
                "name": display_name,
                "name_en": english_name,
                "name_zh": chinese_name,
                "role": role,
                "element": element,
                "appearance_count": appearance_count,
                "avatar_url": f"/api/heroes/{code}/avatar"
                if (PORTRAIT_DIR / code / "c.png").exists()
                else "",
                "element_icon_url": f"/api/icons/elements/{element}.png"
                if element and (ELEMENT_ICON_DIR / f"{element}.png").exists()
                else "",
                "role_icon_url": f"/api/icons/roles/{role}.png"
                if role and (ROLE_ICON_DIR / f"{role}.png").exists()
                else "",
            }
        )
    return heroes


def ensure_recommender_loaded() -> None:
    if transformer_model is None or transformer_encoders is None or not available_heroes:
        load_recommender()


def build_transformer_model_kwargs(encoders: DraftEncoders) -> dict[str, int]:
    kwargs = {
        "num_heroes": encoders.num_heroes,
        "num_roles": encoders.num_roles,
        "num_elements": encoders.num_elements,
        "num_position_buckets": encoders.num_position_buckets,
        "num_warfare_rules": encoders.num_warfare_rules,
    }
    if TRANSFORMER_TRAINING_REPORT_PATH_WARFARE.exists():
        with TRANSFORMER_TRAINING_REPORT_PATH_WARFARE.open(encoding="utf-8") as f:
            config = json.load(f).get("config", {})
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


def load_transformer_weights(model: tf.keras.Model, weights_path: Path) -> None:
    """Load weights saved from Keras 3 training (layers/embedding...) on Linux and Windows."""
    path_str = str(weights_path)
    errors: list[str] = []
    for by_name in (False, True):
        try:
            model.load_weights(path_str, by_name=by_name)
            logging.info("Loaded transformer weights from %s (by_name=%s)", weights_path, by_name)
            return
        except ValueError as exc:
            errors.append(f"by_name={by_name}: {exc}")
    raise RuntimeError(
        f"Failed to load transformer weights from {weights_path}. "
        f"Attempts: {'; '.join(errors)}"
    )


def load_transformer_recommender() -> None:
    global transformer_model, transformer_encoders

    validate_transformer_artifact_paths(
        model_path=TRANSFORMER_MODEL_PATH,
        variables_path=TRANSFORMER_VARIABLES_PATH,
    )
    encoders = DraftEncoders.load(TRANSFORMER_VARIABLES_PATH)
    validate_loaded_encoders(encoders)
    model = build_transformer_draft_model(**build_transformer_model_kwargs(encoders))
    load_transformer_weights(model, TRANSFORMER_MODEL_PATH)
    transformer_encoders = encoders
    transformer_model = model
    logging.info("Loaded Transformer recommender from %s", TRANSFORMER_MODEL_PATH)


def load_recommender() -> None:
    global available_heroes
    global transformer_model, transformer_encoders

    clear_recommendation_cache()
    transformer_model = None
    transformer_encoders = None
    available_heroes = {}

    load_transformer_recommender()
    available_heroes = {
        code: None
        for code in transformer_encoders.hero_to_id
        if code not in {"<PAD>", "<UNK>"}
    }
    if not available_heroes:
        raise RuntimeError("Transformer encoders loaded zero playable heroes")
    # Warm up TensorFlow so the first real request is less likely to stutter.
    predict_next_hero(["unknown"], ["unknown"], "My Team")


def normalize_picks(picks: list[str]) -> list[str]:
    return [pick if pick in available_heroes else "unknown" for pick in picks]


def unavailable_heroes_from_draft(
    ally_picks: list[str],
    enemy_picks: list[str],
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


def final_ban_recommendation(
    *,
    user_team_picks: list[str],
    enemy_team_picks: list[str],
    first_pick_team: str,
    ally_preban: list[str] | None,
    enemy_preban: list[str] | None,
    top_k: int = 4,
) -> dict[str, object]:
    return recommend_final_bans_from_lists(
        user_team_picks,
        enemy_team_picks,
        first_pick_team,
        ally_preban=ally_preban,
        enemy_preban=enemy_preban,
        valid_heroes=set(available_heroes.keys()),
        top_k=top_k,
    )


def first_pick_recommendation(
    *,
    first_pick_team: str,
    user_team_picks: list[str],
    enemy_team_picks: list[str],
    ally_preban: list[str] | None,
    enemy_preban: list[str] | None,
    unavailable_heroes: set[str],
) -> dict[str, object]:
    first_pick_side = parse_first_pick_side(first_pick_team)
    if first_pick_side is None:
        return {
            "phase": "pick",
            "handled_by": "first_pick_stats",
            "first_pick_fallback_level": 4,
            "top_10_heroes": [],
            "top_10_rates": [],
        }
    return recommend_first_pick(
        ally_preban=ally_preban,
        enemy_preban=enemy_preban,
        first_pick_side=first_pick_side,
        excluded_heroes=unavailable_heroes,
        top_k=10,
    )


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


def predict_transformer_hero_probs(
    arrays: dict[str, np.ndarray],
    *,
    warfare_rules: str,
    inference_debug: dict[str, object] | None = None,
) -> np.ndarray:
    if transformer_model is None or transformer_encoders is None:
        raise RuntimeError("Transformer recommender is not loaded")

    normalized_rule = normalize_warfare_rules_param(warfare_rules)
    candidate_mask = arrays["candidate_mask"][0]
    if inference_debug is not None:
        inference_debug["normalized_warfare_rules"] = normalized_rule
        inference_debug["any_blending_used"] = normalized_rule == WARFARE_RULE_ANY
        inference_debug["concrete_rule_inference_calls"] = []

    if normalized_rule != WARFARE_RULE_ANY:
        if inference_debug is not None:
            inference_debug["concrete_rule_inference_calls"].append(normalized_rule)
        rule_arrays = clone_inference_arrays_with_rule(
            arrays,
            transformer_encoders,
            normalized_rule,
        )
        probs = transformer_model.predict(arrays_to_model_inputs(rule_arrays), verbose=0)[0]
        return mask_and_normalize_probs(probs, candidate_mask)

    priors = transformer_encoders.warfare_rule_priors or compute_warfare_rule_priors(
        {rule: 1 for rule in WARFARE_RULE_ORDER}
    )
    if inference_debug is not None:
        inference_debug["warfare_rule_priors"] = dict(priors)
    probs_by_rule: dict[str, np.ndarray] = {}
    for rule in WARFARE_RULE_ORDER:
        if inference_debug is not None:
            inference_debug["concrete_rule_inference_calls"].append(rule)
        rule_arrays = clone_inference_arrays_with_rule(arrays, transformer_encoders, rule)
        raw_probs = transformer_model.predict(arrays_to_model_inputs(rule_arrays), verbose=0)[0]
        probs_by_rule[rule] = mask_and_normalize_probs(raw_probs, candidate_mask)

    blended = blend_concrete_rule_probabilities(probs_by_rule, priors)
    return mask_and_normalize_probs(blended, candidate_mask)


def predict_next_hero_transformer(
    enemy_team_picks: list[str],
    user_team_picks: list[str],
    first_pick_team: str,
    *,
    ally_preban: list[str] | None = None,
    enemy_preban: list[str] | None = None,
    warfare_rules: str = WARFARE_RULE_ANY,
    inference_debug: dict[str, object] | None = None,
):
    if transformer_model is None or transformer_encoders is None:
        raise RuntimeError("Transformer recommender is not loaded")

    user_team_picks = normalize_picks(user_team_picks)
    enemy_team_picks = normalize_picks(enemy_team_picks)
    ally_preban = normalize_picks(ally_preban or [])
    enemy_preban = normalize_picks(enemy_preban or [])
    unavailable_heroes = unavailable_heroes_from_draft(
        user_team_picks,
        enemy_team_picks,
        ally_preban,
        enemy_preban,
    )

    if is_first_pick_recommendation_turn(first_pick_team, user_team_picks, enemy_team_picks):
        return first_pick_recommendation(
            first_pick_team=first_pick_team,
            user_team_picks=user_team_picks,
            enemy_team_picks=enemy_team_picks,
            ally_preban=ally_preban,
            enemy_preban=enemy_preban,
            unavailable_heroes=unavailable_heroes,
        )

    arrays = build_inference_arrays(
        transformer_encoders,
        user_picks=user_team_picks,
        enemy_picks=enemy_team_picks,
        first_pick_team=first_pick_team,
        ally_preban=ally_preban,
        enemy_preban=enemy_preban,
        warfare_rules=warfare_rules,
    )
    hero_probs = predict_transformer_hero_probs(
        arrays,
        warfare_rules=warfare_rules,
        inference_debug=inference_debug,
    )

    candidate_mask = arrays["candidate_mask"][0]
    top_10_heroes, top_10_rates = rank_available_recommendations(
        hero_probs,
        id_to_hero=transformer_encoders.id_to_hero,
        hero_to_id=transformer_encoders.hero_to_id,
        unavailable_heroes=unavailable_heroes,
        candidate_mask=candidate_mask,
        k=10,
    )
    reranked = maybe_rerank_transformer_recommendations(
        top_10_heroes=top_10_heroes,
        top_10_rates=top_10_rates,
        hero_probs=hero_probs,
        arrays=arrays,
        user_team_picks=user_team_picks,
        enemy_team_picks=enemy_team_picks,
        unavailable_heroes=unavailable_heroes,
        first_pick_team=first_pick_team,
    )
    top_10_heroes = reranked["top_10_heroes"]
    top_10_rates = reranked["top_10_rates"]

    if len(user_team_picks) >= 5 and len(enemy_team_picks) >= 5:
        return final_ban_recommendation(
            user_team_picks=user_team_picks,
            enemy_team_picks=enemy_team_picks,
            first_pick_team=first_pick_team,
            ally_preban=ally_preban,
            enemy_preban=enemy_preban,
            top_k=4,
        )

    pick_payload: dict[str, object] = {
        "top_10_heroes": top_10_heroes,
        "top_10_rates": top_10_rates,
        "phase": "pick",
    }
    if reranker_enabled():
        pick_payload["reranker_enabled"] = reranked.get("reranker_enabled", True)
        pick_payload["reranker_applied"] = reranked.get("reranker_applied", False)
    if debug_recommendations_enabled() and "debug_recommendations" in reranked:
        pick_payload["debug_recommendations"] = reranked["debug_recommendations"]
    return pick_payload


def predict_next_hero(
    enemy_team_picks: list[str],
    user_team_picks: list[str],
    first_pick_team: str,
    *,
    ally_preban: list[str] | None = None,
    enemy_preban: list[str] | None = None,
    warfare_rules: str = WARFARE_RULE_ANY,
    inference_debug: dict[str, object] | None = None,
):
    return predict_next_hero_transformer(
        enemy_team_picks,
        user_team_picks,
        first_pick_team,
        ally_preban=ally_preban,
        enemy_preban=enemy_preban,
        warfare_rules=warfare_rules,
        inference_debug=inference_debug,
    )


@app.get("/api/heroes")
def heroes():
    try:
        return jsonify({"heroes": load_hero_options()}), 200
    except Exception as e:
        logging.exception("Error loading heroes")
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.get("/api/heroes/<hero_code>/avatar")
def hero_avatar(hero_code: str):
    portrait_path = PORTRAIT_DIR / hero_code / "c.png"
    if not portrait_path.exists():
        abort(404)

    return send_from_directory(portrait_path.parent, portrait_path.name)


@app.get("/api/icons/elements/<icon_name>")
def element_icon(icon_name: str):
    icon_path = ELEMENT_ICON_DIR / icon_name
    if not icon_path.exists():
        abort(404)

    return send_from_directory(ELEMENT_ICON_DIR, icon_name)


@app.get("/api/icons/roles/<icon_name>")
def role_icon(icon_name: str):
    icon_path = ROLE_ICON_DIR / icon_name
    if not icon_path.exists():
        abort(404)

    return send_from_directory(ROLE_ICON_DIR, icon_name)


@app.get("/api/init_recommender")
@app.get("/init_recommender")
def init_recommender():
    try:
        load_recommender()
        return jsonify({"message": "Recommender model initialized successfully"}), 200
    except Exception as e:
        logging.exception("Error initializing recommender")
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.get("/api/recommend")
@app.get("/recommend")
def recommend_characters():
    try:
        ensure_recommender_loaded()
        enemy_picks = parse_picks(request.args.get("enemy_picks"))
        user_picks = parse_picks(request.args.get("user_picks"))
        ally_preban = parse_picks(request.args.get("ally_preban"))
        enemy_preban = parse_picks(request.args.get("enemy_preban"))
        first_pick_team = request.args.get("first_pick_team", "My Team")
        raw_warfare_rules = request.args.get("warfare_rules")

        if first_pick_team not in {"My Team", "Enemy Team"}:
            return jsonify({"message": "first_pick_team must be 'My Team' or 'Enemy Team'"}), 400

        try:
            warfare_rules = normalize_warfare_rules_param(raw_warfare_rules)
        except ValueError as exc:
            return jsonify({"message": str(exc)}), 400

        normalized_user_picks = normalize_picks(user_picks)
        normalized_enemy_picks = normalize_picks(enemy_picks)
        normalized_ally_preban = normalize_picks(ally_preban)
        normalized_enemy_preban = normalize_picks(enemy_preban)

        cache_key = recommendation_cache_key(
            user_picks=normalized_user_picks,
            enemy_picks=normalized_enemy_picks,
            ally_preban=normalized_ally_preban,
            enemy_preban=normalized_enemy_preban,
            first_pick_team=first_pick_team,
            warfare_rules=warfare_rules,
        )
        cached_payload = get_cached_recommendation(cache_key)
        from_cache = cached_payload is not None
        inference_debug: dict[str, object] | None = {} if debug_recommendations_enabled() else None

        if from_cache:
            payload = dict(cached_payload)
        else:
            payload = dict(
                predict_next_hero(
                    normalized_enemy_picks,
                    normalized_user_picks,
                    first_pick_team,
                    ally_preban=normalized_ally_preban,
                    enemy_preban=normalized_enemy_preban,
                    warfare_rules=warfare_rules,
                    inference_debug=inference_debug,
                )
            )
            store_cached_recommendation(cache_key, payload)

        if debug_recommendations_enabled():
            payload["debug"] = collect_recommend_debug(
                raw_warfare_rules=raw_warfare_rules,
                normalized_warfare_rules=warfare_rules,
                user_picks=normalized_user_picks,
                enemy_picks=normalized_enemy_picks,
                ally_preban=normalized_ally_preban,
                enemy_preban=normalized_enemy_preban,
                first_pick_team=first_pick_team,
                from_cache=from_cache,
                payload=payload,
                inference_debug=inference_debug,
                available_heroes=available_heroes,
                reranker_enabled=reranker_enabled(),
                model_path=TRANSFORMER_MODEL_PATH,
                variables_path=TRANSFORMER_VARIABLES_PATH,
            )
            payload["debug"]["cache_key"] = {
                "user_picks": list(cache_key[0]),
                "enemy_picks": list(cache_key[1]),
                "ally_preban": list(cache_key[2]),
                "enemy_preban": list(cache_key[3]),
                "first_pick_team": cache_key[4],
                "warfare_rules": cache_key[5],
            }

        return jsonify(payload), 200
    except Exception as e:
        logging.exception("Error generating recommendation")
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.get("/api/preban_recommend")
@app.get("/preban_recommend")
def recommend_preban_characters():
    try:
        excluded_heroes = parse_picks(request.args.get("excluded_heroes"))
        top_k = parse_top_k(request.args.get("top_k"), default=10, maximum=50)
        raw_side = (request.args.get("preban_side") or "").strip().lower()
        if raw_side == "user":
            source = "ally"
        elif raw_side == "enemy":
            source = "enemy"
        else:
            source = "combined"
        first_pick_side = resolve_preban_first_pick_side(
            parse_first_pick_side(request.args.get("first_pick_team")),
            preban_source=source,
        )
        return jsonify(
            recommend_prebans(
                excluded_heroes,
                top_k,
                source=source,
                first_pick_side=first_pick_side,
            )
        ), 200
    except Exception as e:
        logging.exception("Error generating preban recommendation")
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.get("/api/debug/runtime")
def debug_runtime():
    try:
        ensure_recommender_loaded()
        return jsonify(
            build_runtime_debug_payload(
                recommender_service_file=str(Path(__file__).resolve()),
                transformer_model_path=TRANSFORMER_MODEL_PATH,
                transformer_variables_path=TRANSFORMER_VARIABLES_PATH,
                transformer_model_loaded=transformer_model is not None,
                transformer_encoders=transformer_encoders,
                available_heroes=available_heroes,
            )
        ), 200
    except Exception as exc:
        logging.exception("Error building runtime debug payload")
        return jsonify({"message": str(exc)}), 500


@app.get("/api/debug/hero/<hero_code>")
def debug_hero(hero_code: str):
    try:
        ensure_recommender_loaded()
        return jsonify(
            build_hero_debug_payload(
                hero_code,
                transformer_encoders=transformer_encoders,
                available_heroes=available_heroes,
            )
        ), 200
    except Exception as exc:
        logging.exception("Error building hero debug payload for %s", hero_code)
        return jsonify({"message": str(exc)}), 500


@app.get("/api/status")
@app.get("/status")
def status():
    return jsonify(
        {
            "message": "Server is running",
            "app": "e7_bp_helper",
            "frontend_enabled": frontend_enabled(),
            "reranker_enabled": reranker_enabled(),
            "debug_recommendations_enabled": debug_recommendations_enabled(),
        }
    ), 200


def frontend_enabled() -> bool:
    return FRONTEND_DIST_DIR.is_dir() and (FRONTEND_DIST_DIR / "index.html").is_file()


@app.get("/")
def serve_frontend_index():
    if not frontend_enabled():
        abort(404)
    return send_from_directory(FRONTEND_DIST_DIR, "index.html")


@app.get("/<path:asset_path>")
def serve_frontend_asset(asset_path: str):
    if asset_path.startswith("api/"):
        abort(404)

    if not frontend_enabled():
        abort(404)

    requested = FRONTEND_DIST_DIR / asset_path
    if requested.is_file():
        return send_from_directory(FRONTEND_DIST_DIR, asset_path)

    return send_from_directory(FRONTEND_DIST_DIR, "index.html")


def run_server(*, host: str | None = None, port: int | None = None) -> None:
    selected_host = host or os.environ.get("HOST", "127.0.0.1")
    selected_port = port or int(os.environ.get("PORT") or 5000)
    (APP_BASE_DIR / "recommender_service_port.txt").write_text(str(selected_port), encoding="utf-8")
    print(f"Starting recommender service on port {selected_port}")
    print(f"RECOMMENDER_RERANKER={'enabled' if reranker_enabled() else 'disabled'}")
    print(f"RECOMMENDER_DEBUG={'enabled' if debug_recommendations_enabled() else 'disabled'}")
    if frontend_enabled():
        print(f"Serving frontend from {FRONTEND_DIST_DIR}")
    app.run(host=selected_host, port=selected_port, debug=False, use_reloader=False)


if __name__ == "__main__":
    run_server()
