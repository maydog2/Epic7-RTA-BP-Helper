"""Compact Transformer draft predictor model."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from transformer_draft_data import validate_model_input_arrays


def build_transformer_draft_model(
    *,
    num_heroes: int,
    num_roles: int = 2,
    num_elements: int = 2,
    num_position_buckets: int = 9,
    num_warfare_rules: int = 5,
    d_model: int = 128,
    num_heads: int = 4,
    ff_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> tf.keras.Model:
    draft_hero_ids = tf.keras.layers.Input(shape=(10,), name="draft_hero_ids", dtype="int32")
    draft_role_ids = tf.keras.layers.Input(shape=(10,), name="draft_role_ids", dtype="int32")
    draft_element_ids = tf.keras.layers.Input(shape=(10,), name="draft_element_ids", dtype="int32")
    draft_side_ids = tf.keras.layers.Input(shape=(10,), name="draft_side_ids", dtype="int32")
    draft_order_ids = tf.keras.layers.Input(shape=(10,), name="draft_order_ids", dtype="int32")
    draft_position_bucket_ids = tf.keras.layers.Input(
        shape=(10,),
        name="draft_position_bucket_ids",
        dtype="int32",
    )
    is_first_pick_slot = tf.keras.layers.Input(shape=(10,), name="is_first_pick_slot", dtype="int32")
    ally_preban_ids = tf.keras.layers.Input(shape=(2,), name="ally_preban_ids", dtype="int32")
    ally_preban_role_ids = tf.keras.layers.Input(shape=(2,), name="ally_preban_role_ids", dtype="int32")
    ally_preban_element_ids = tf.keras.layers.Input(shape=(2,), name="ally_preban_element_ids", dtype="int32")
    enemy_preban_ids = tf.keras.layers.Input(shape=(2,), name="enemy_preban_ids", dtype="int32")
    enemy_preban_role_ids = tf.keras.layers.Input(shape=(2,), name="enemy_preban_role_ids", dtype="int32")
    enemy_preban_element_ids = tf.keras.layers.Input(shape=(2,), name="enemy_preban_element_ids", dtype="int32")
    draft_filled_mask = tf.keras.layers.Input(shape=(10,), name="draft_filled_mask", dtype="float32")
    candidate_mask = tf.keras.layers.Input(shape=(num_heroes,), name="candidate_mask", dtype="float32")
    current_position_bucket_id = tf.keras.layers.Input(
        shape=(1,),
        name="current_position_bucket_id",
        dtype="int32",
    )
    current_order = tf.keras.layers.Input(shape=(1,), name="current_order", dtype="int32")
    current_side_id = tf.keras.layers.Input(shape=(1,), name="current_side_id", dtype="int32")
    warfare_rule_id = tf.keras.layers.Input(shape=(1,), name="warfare_rule_id", dtype="int32")

    hero_embedding = tf.keras.layers.Embedding(num_heroes, d_model, name="hero_embedding")
    role_embedding = tf.keras.layers.Embedding(num_roles, d_model, name="role_embedding")
    element_embedding = tf.keras.layers.Embedding(num_elements, d_model, name="element_embedding")
    side_embedding = tf.keras.layers.Embedding(3, d_model, name="side_embedding")
    order_embedding = tf.keras.layers.Embedding(11, d_model, name="order_embedding")
    position_bucket_embedding = tf.keras.layers.Embedding(
        num_position_buckets,
        d_model,
        name="position_bucket_embedding",
    )
    slot_flag_embedding = tf.keras.layers.Embedding(2, d_model, name="slot_flag_embedding")
    warfare_rule_embedding = tf.keras.layers.Embedding(
        num_warfare_rules,
        d_model,
        name="warfare_rule_embedding",
    )

    hero_tokens = hero_embedding(draft_hero_ids)
    role_tokens = role_embedding(draft_role_ids)
    element_tokens = element_embedding(draft_element_ids)
    side_tokens = side_embedding(draft_side_ids)
    order_tokens = order_embedding(draft_order_ids)
    bucket_tokens = position_bucket_embedding(draft_position_bucket_ids)
    slot_tokens = slot_flag_embedding(is_first_pick_slot)

    token_features = tf.keras.layers.Add()(
        [hero_tokens, role_tokens, element_tokens, side_tokens, order_tokens, bucket_tokens, slot_tokens]
    )
    token_features = tf.keras.layers.LayerNormalization(name="token_norm")(token_features)

    filled = tf.keras.layers.Lambda(
        lambda x: tf.cast(x, tf.float32),
        name="draft_filled_float",
    )(draft_filled_mask)
    pairwise_mask = tf.keras.layers.Lambda(
        lambda x: tf.cast(x[:, :, tf.newaxis] * x[:, tf.newaxis, :], tf.bool),
        name="pairwise_attention_mask",
    )(filled)

    x = token_features
    for idx in range(num_layers):
        attn_out = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name=f"mha_{idx}",
        )(x, x, attention_mask=pairwise_mask)
        x = tf.keras.layers.LayerNormalization(name=f"norm1_{idx}")(x + attn_out)
        ff = tf.keras.layers.Dense(ff_dim, activation="relu", name=f"ff1_{idx}")(x)
        ff = tf.keras.layers.Dropout(dropout, name=f"ff_drop_{idx}")(ff)
        ff = tf.keras.layers.Dense(d_model, name=f"ff2_{idx}")(ff)
        x = tf.keras.layers.LayerNormalization(name=f"norm2_{idx}")(x + ff)

    masked_tokens = tf.keras.layers.Multiply(name="masked_tokens")(
        [x, tf.expand_dims(filled, axis=-1)]
    )
    token_count = tf.keras.layers.Lambda(
        lambda t: tf.maximum(tf.reduce_sum(t, axis=1, keepdims=True), 1.0),
        name="token_count",
    )(filled)
    draft_context = tf.keras.layers.Lambda(
        lambda inputs: tf.reduce_sum(inputs[0], axis=1) / inputs[1],
        name="draft_context_pool",
    )([masked_tokens, token_count])

    current_pick_context = tf.keras.layers.Add(name="current_pick_feature_sum")(
        [
            position_bucket_embedding(current_position_bucket_id),
            order_embedding(current_order),
            side_embedding(current_side_id),
        ]
    )
    current_pick_context = tf.keras.layers.Flatten(name="current_pick_context_flat")(current_pick_context)

    ally_preban_tokens = tf.keras.layers.Add(name="ally_preban_feature_sum")(
        [
            hero_embedding(ally_preban_ids),
            role_embedding(ally_preban_role_ids),
            element_embedding(ally_preban_element_ids),
        ]
    )
    enemy_preban_tokens = tf.keras.layers.Add(name="enemy_preban_feature_sum")(
        [
            hero_embedding(enemy_preban_ids),
            role_embedding(enemy_preban_role_ids),
            element_embedding(enemy_preban_element_ids),
        ]
    )
    ally_preban_context = tf.reduce_mean(ally_preban_tokens, axis=1)
    enemy_preban_context = tf.reduce_mean(enemy_preban_tokens, axis=1)
    warfare_rule_context = tf.keras.layers.Flatten(name="warfare_rule_context_flat")(
        warfare_rule_embedding(warfare_rule_id)
    )

    context = tf.keras.layers.Concatenate(name="context_concat")(
        [draft_context, current_pick_context, ally_preban_context, enemy_preban_context, warfare_rule_context]
    )
    context = tf.keras.layers.Dense(d_model, activation="relu", name="context_proj")(context)
    context = tf.keras.layers.Dropout(dropout, name="context_drop")(context)

    next_hero_logits = tf.keras.layers.Dense(num_heroes, name="next_hero_logits")(context)
    masked_logits = tf.keras.layers.Lambda(
        lambda inputs: inputs[0] + (1.0 - tf.cast(inputs[1], tf.float32)) * -1e9,
        name="masked_next_hero_logits",
    )([next_hero_logits, candidate_mask])
    next_hero_probs = tf.keras.layers.Activation("softmax", name="next_hero")(masked_logits)

    return tf.keras.Model(
        inputs=[
            draft_hero_ids,
            draft_role_ids,
            draft_element_ids,
            draft_side_ids,
            draft_order_ids,
            draft_position_bucket_ids,
            is_first_pick_slot,
            ally_preban_ids,
            ally_preban_role_ids,
            ally_preban_element_ids,
            enemy_preban_ids,
            enemy_preban_role_ids,
            enemy_preban_element_ids,
            draft_filled_mask,
            candidate_mask,
            current_position_bucket_id,
            current_order,
            current_side_id,
            warfare_rule_id,
        ],
        outputs=next_hero_probs,
        name="transformer_draft_model",
    )


def arrays_to_model_inputs(arrays: dict, *, split_name: str = "dataset") -> list:
    validate_model_input_arrays(arrays, split_name=split_name)
    draft_filled_mask = (arrays["draft_hero_ids"] > 0).astype("float32")
    current_position_bucket_id = arrays["current_position_bucket_id"]
    current_order = arrays["current_order"]
    current_side_id = arrays["current_side_id"]
    warfare_rule_id = arrays["warfare_rule_id"]
    if current_position_bucket_id.ndim == 1:
        current_position_bucket_id = current_position_bucket_id[:, np.newaxis]
        current_order = current_order[:, np.newaxis]
        current_side_id = current_side_id[:, np.newaxis]
        warfare_rule_id = warfare_rule_id[:, np.newaxis]

    return [
        arrays["draft_hero_ids"],
        arrays["draft_role_ids"],
        arrays["draft_element_ids"],
        arrays["draft_side_ids"],
        arrays["draft_order_ids"],
        arrays["draft_position_bucket_ids"],
        arrays["is_first_pick_slot"],
        arrays["ally_preban_ids"],
        arrays["ally_preban_role_ids"],
        arrays["ally_preban_element_ids"],
        arrays["enemy_preban_ids"],
        arrays["enemy_preban_role_ids"],
        arrays["enemy_preban_element_ids"],
        draft_filled_mask,
        arrays["candidate_mask"],
        current_position_bucket_id,
        current_order,
        current_side_id,
        warfare_rule_id,
    ]
