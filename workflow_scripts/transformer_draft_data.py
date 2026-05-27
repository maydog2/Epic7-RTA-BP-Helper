"""Shared draft-state dataset utilities for the Transformer pick model."""

from __future__ import annotations

import json
import pickle
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runtime_paths import DATA_DIR, HERO_DETAILS_PATH, TRANSFORMER_DATA_DIR
from match_history_utils import RAW_JSONL_PATH, get_position_bucket, validate_match_record

PROJECT_ROOT = _PROJECT_ROOT
TRANSFORMER_TRAIN_NPZ = TRANSFORMER_DATA_DIR / "draft_train.npz"
TRANSFORMER_VAL_NPZ = TRANSFORMER_DATA_DIR / "draft_val.npz"
TRANSFORMER_ENCODERS_PATH = TRANSFORMER_DATA_DIR / "rec_variables.pkl"
TRANSFORMER_WEIGHTS_PATH = TRANSFORMER_DATA_DIR / "rec_model.weights.h5"
TRANSFORMER_TRAINING_REPORT_PATH = TRANSFORMER_DATA_DIR / "training_report.json"
TRANSFORMER_TRAINING_LOG_PATH = TRANSFORMER_DATA_DIR / "training_log.csv"
TRANSFORMER_EVAL_REPORT_PATH = TRANSFORMER_DATA_DIR / "eval_report.json"
TRANSFORMER_SPLIT_PATH = TRANSFORMER_DATA_DIR / "split.json"
TRANSFORMER_TENSORBOARD_DIR = TRANSFORMER_DATA_DIR / "tensorboard"
TRANSFORMER_WEIGHTS_PATH_WARFARE = TRANSFORMER_DATA_DIR / "rec_model_position_warfare.weights.h5"
TRANSFORMER_TRAINING_REPORT_PATH_WARFARE = TRANSFORMER_DATA_DIR / "training_report_position_warfare.json"
TRANSFORMER_TRAINING_LOG_PATH_WARFARE = TRANSFORMER_DATA_DIR / "training_log_position_warfare.csv"
TRANSFORMER_TENSORBOARD_DIR_WARFARE = TRANSFORMER_DATA_DIR / "tensorboard_position_warfare"
TRANSFORMER_EVAL_REPORT_PATH_WARFARE = TRANSFORMER_DATA_DIR / "eval_report_position_warfare.json"

DRAFT_SLOTS = 10
MAX_PREBAN = 2
PAD_HERO_ID = 0
UNK_HERO_ID = 1
PAD_META_ID = 0
UNK_META_ID = 1
PAD_SIDE_ID = 0
ALLY_SIDE_ID = 1
ENEMY_SIDE_ID = 2

# RTA pick order by draft slot (1..10), 0-indexed slot -> side id
SIDE_ORDER_ALLY_FIRST = np.array(
    [ALLY_SIDE_ID, ENEMY_SIDE_ID, ENEMY_SIDE_ID, ALLY_SIDE_ID, ALLY_SIDE_ID,
     ENEMY_SIDE_ID, ENEMY_SIDE_ID, ALLY_SIDE_ID, ALLY_SIDE_ID, ENEMY_SIDE_ID],
    dtype=np.int32,
)
SIDE_ORDER_ENEMY_FIRST = np.array(
    [ENEMY_SIDE_ID, ALLY_SIDE_ID, ALLY_SIDE_ID, ENEMY_SIDE_ID, ENEMY_SIDE_ID,
     ALLY_SIDE_ID, ALLY_SIDE_ID, ENEMY_SIDE_ID, ENEMY_SIDE_ID, ALLY_SIDE_ID],
    dtype=np.int32,
)

PAD_POSITION_BUCKET_ID = 0
POSITION_BUCKET_ORDER = (
    "1",
    "2_3",
    "4",
    "5_protected",
    "6_protected",
    "7",
    "8_9",
    "10",
)

WARFARE_RULE_ORDER = ("Support", "Offense", "Defense", "Resistance")
CONCRETE_WARFARE_RULES = WARFARE_RULE_ORDER
WARFARE_RULE_ANY = "ANY"
PAD_WARFARE_RULE_ID = 0

REQUIRED_MODEL_INPUT_ARRAYS = (
    "draft_hero_ids",
    "draft_role_ids",
    "draft_element_ids",
    "draft_side_ids",
    "draft_order_ids",
    "draft_position_bucket_ids",
    "is_first_pick_slot",
    "ally_preban_ids",
    "ally_preban_role_ids",
    "ally_preban_element_ids",
    "enemy_preban_ids",
    "enemy_preban_role_ids",
    "enemy_preban_element_ids",
    "candidate_mask",
    "current_position_bucket_id",
    "current_order",
    "current_side_id",
    "warfare_rule_id",
)

REQUIRED_NPZ_ARRAYS = REQUIRED_MODEL_INPUT_ARRAYS + (
    "target_next_hero",
)


@dataclass
class DraftEncoders:
    hero_to_id: dict[str, int]
    id_to_hero: list[str]
    num_heroes: int
    role_to_id: dict[str, int]
    element_to_id: dict[str, int]
    hero_role_ids: np.ndarray
    hero_element_ids: np.ndarray
    position_bucket_to_id: dict[str, int] | None = None
    id_to_position_bucket: list[str] | None = None
    warfare_rule_to_id: dict[str, int] | None = None
    id_to_warfare_rule: list[str] | None = None
    warfare_rule_priors: dict[str, float] | None = None

    def hero_id(self, code: str | None) -> int:
        if not code:
            return PAD_HERO_ID
        return self.hero_to_id.get(code, UNK_HERO_ID)

    @property
    def num_roles(self) -> int:
        return len(self.role_to_id)

    @property
    def num_elements(self) -> int:
        return len(self.element_to_id)

    @property
    def num_position_buckets(self) -> int:
        self._require_position_bucket_encoders()
        return len(self.position_bucket_to_id)  # type: ignore[arg-type]

    def _require_position_bucket_encoders(self) -> None:
        if not self.position_bucket_to_id or not self.id_to_position_bucket:
            raise ValueError(
                "DraftEncoders is missing position_bucket mappings. "
                "Rebuild with: python workflow_scripts/build_transformer_draft_dataset.py"
            )

    def _require_warfare_rule_encoders(self) -> None:
        if not self.warfare_rule_to_id or not self.id_to_warfare_rule:
            raise ValueError(
                "DraftEncoders is missing warfare_rule mappings. "
                "Rebuild with: python workflow_scripts/build_transformer_draft_dataset.py"
            )

    @property
    def num_warfare_rules(self) -> int:
        self._require_warfare_rule_encoders()
        return len(self.warfare_rule_to_id)  # type: ignore[arg-type]

    def warfare_rule_id_for_name(self, rule_name: str) -> int:
        self._require_warfare_rule_encoders()
        rule_id = self.warfare_rule_to_id.get(rule_name)  # type: ignore[union-attr]
        if rule_id is None:
            raise ValueError(f"Unknown concrete warfare rule: {rule_name!r}")
        return int(rule_id)

    def position_bucket_id_for_order(self, order: int) -> int:
        self._require_position_bucket_encoders()
        bucket = get_position_bucket(order)
        return self.position_bucket_to_id[bucket]  # type: ignore[index]

    def draft_position_bucket_ids(self) -> np.ndarray:
        return np.array(
            [self.position_bucket_id_for_order(order) for order in range(1, DRAFT_SLOTS + 1)],
            dtype=np.int32,
        )

    def role_ids_for_hero_ids(self, hero_ids: np.ndarray) -> np.ndarray:
        clipped = np.clip(hero_ids, 0, len(self.hero_role_ids) - 1)
        return self.hero_role_ids[clipped].astype(np.int32)

    def element_ids_for_hero_ids(self, hero_ids: np.ndarray) -> np.ndarray:
        clipped = np.clip(hero_ids, 0, len(self.hero_element_ids) - 1)
        return self.hero_element_ids[clipped].astype(np.int32)

    def save(self, path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> DraftEncoders:
        with path.open("rb") as f:
            return pickle.load(f)


@dataclass
class DraftSample:
    draft_hero_ids: np.ndarray
    draft_role_ids: np.ndarray
    draft_element_ids: np.ndarray
    draft_side_ids: np.ndarray
    draft_order_ids: np.ndarray
    is_first_pick_slot: np.ndarray
    ally_preban_ids: np.ndarray
    ally_preban_role_ids: np.ndarray
    ally_preban_element_ids: np.ndarray
    enemy_preban_ids: np.ndarray
    enemy_preban_role_ids: np.ndarray
    enemy_preban_element_ids: np.ndarray
    candidate_mask: np.ndarray
    first_pick_side: int
    draft_step: int
    match_id: int
    target_next_hero: int
    target_next_side: int
    draft_position_bucket_ids: np.ndarray
    current_position_bucket_id: int
    current_order: int
    current_side_id: int
    warfare_rule_id: int


def build_warfare_rule_vocab() -> tuple[dict[str, int], list[str]]:
    warfare_rule_to_id = {"<PAD>": PAD_WARFARE_RULE_ID}
    for rule in WARFARE_RULE_ORDER:
        warfare_rule_to_id[rule] = len(warfare_rule_to_id)
    id_to_warfare_rule = [""] * len(warfare_rule_to_id)
    for rule, idx in warfare_rule_to_id.items():
        id_to_warfare_rule[idx] = rule
    return warfare_rule_to_id, id_to_warfare_rule


def build_position_bucket_vocab() -> tuple[dict[str, int], list[str]]:
    position_bucket_to_id = {"<PAD>": PAD_POSITION_BUCKET_ID}
    for bucket in POSITION_BUCKET_ORDER:
        position_bucket_to_id[bucket] = len(position_bucket_to_id)
    id_to_position_bucket = [""] * len(position_bucket_to_id)
    for bucket, idx in position_bucket_to_id.items():
        id_to_position_bucket[idx] = bucket
    return position_bucket_to_id, id_to_position_bucket


def validate_npz_arrays(arrays: dict[str, np.ndarray], *, split_name: str) -> None:
    missing = [key for key in REQUIRED_NPZ_ARRAYS if key not in arrays]
    if missing:
        raise ValueError(
            f"{split_name} NPZ is missing required arrays: {missing}. "
            "Rebuild with: python workflow_scripts/build_transformer_draft_dataset.py"
        )


def validate_model_input_arrays(arrays: dict[str, np.ndarray], *, split_name: str = "inference") -> None:
    missing = [key for key in REQUIRED_MODEL_INPUT_ARRAYS if key not in arrays]
    if missing:
        raise ValueError(
            f"{split_name} input is missing required arrays: {missing}. "
            "Rebuild encoders/dataset or update inference array builder."
        )


def load_hero_metadata(path: Path = HERO_DETAILS_PATH) -> dict[str, tuple[str, str]]:
    metadata: dict[str, tuple[str, str]] = {}
    if not path.exists():
        return metadata
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = (row.get("Hero") or "").strip()
            if not code:
                continue
            role = (row.get("Role") or "unknown").strip().lower() or "unknown"
            element = (row.get("Element") or "unknown").strip().lower() or "unknown"
            metadata[code] = (role, element)
    return metadata


def build_hero_vocab(matches: list[dict[str, Any]]) -> DraftEncoders:
    heroes: set[str] = set()
    for match in matches:
        for field in ("ally_preban", "enemy_preban"):
            heroes.update(match.get(field) or [])
        for entry in match.get("draft") or []:
            if entry.get("hero"):
                heroes.add(entry["hero"])

    hero_to_id = {"<PAD>": PAD_HERO_ID, "<UNK>": UNK_HERO_ID}
    for code in sorted(heroes):
        hero_to_id[code] = len(hero_to_id)

    id_to_hero = [""] * len(hero_to_id)
    for code, idx in hero_to_id.items():
        id_to_hero[idx] = code

    metadata = load_hero_metadata()
    roles = sorted({metadata.get(code, ("unknown", "unknown"))[0] for code in heroes})
    elements = sorted({metadata.get(code, ("unknown", "unknown"))[1] for code in heroes})
    role_to_id = {"<PAD>": PAD_META_ID, "<UNK>": UNK_META_ID}
    element_to_id = {"<PAD>": PAD_META_ID, "<UNK>": UNK_META_ID}
    for role in roles:
        if role != "unknown":
            role_to_id[role] = len(role_to_id)
    for element in elements:
        if element != "unknown":
            element_to_id[element] = len(element_to_id)

    hero_role_ids = np.full(len(hero_to_id), UNK_META_ID, dtype=np.int32)
    hero_element_ids = np.full(len(hero_to_id), UNK_META_ID, dtype=np.int32)
    hero_role_ids[PAD_HERO_ID] = PAD_META_ID
    hero_element_ids[PAD_HERO_ID] = PAD_META_ID
    hero_role_ids[UNK_HERO_ID] = UNK_META_ID
    hero_element_ids[UNK_HERO_ID] = UNK_META_ID
    for code, hero_id in hero_to_id.items():
        if code in {"<PAD>", "<UNK>"}:
            continue
        role, element = metadata.get(code, ("unknown", "unknown"))
        hero_role_ids[hero_id] = role_to_id.get(role, UNK_META_ID)
        hero_element_ids[hero_id] = element_to_id.get(element, UNK_META_ID)

    position_bucket_to_id, id_to_position_bucket = build_position_bucket_vocab()
    warfare_rule_to_id, id_to_warfare_rule = build_warfare_rule_vocab()

    return DraftEncoders(
        hero_to_id=hero_to_id,
        id_to_hero=id_to_hero,
        num_heroes=len(hero_to_id),
        role_to_id=role_to_id,
        element_to_id=element_to_id,
        hero_role_ids=hero_role_ids,
        hero_element_ids=hero_element_ids,
        position_bucket_to_id=position_bucket_to_id,
        id_to_position_bucket=id_to_position_bucket,
        warfare_rule_to_id=warfare_rule_to_id,
        id_to_warfare_rule=id_to_warfare_rule,
        warfare_rule_priors=None,
    )


def extract_concrete_warfare_rule(match: dict[str, Any]) -> str | None:
    raw_value = match.get("warfare_rules")
    if raw_value is None:
        return None
    if isinstance(raw_value, dict):
        raw_value = raw_value.get("warfare_rules")
    text = str(raw_value).strip()
    if not text:
        return None
    for rule in WARFARE_RULE_ORDER:
        if text == rule:
            return rule
    return None


def load_transformer_training_matches(
    raw_path: Path = RAW_JSONL_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load structurally valid matches that also have a concrete warfare rule."""
    raw_matches = load_valid_matches(raw_path)
    kept: list[dict[str, Any]] = []
    skipped_missing_or_empty = 0
    skipped_invalid = 0
    per_rule_match_count = {rule: 0 for rule in WARFARE_RULE_ORDER}

    for match in raw_matches:
        rule = extract_concrete_warfare_rule(match)
        if rule is None:
            raw_value = match.get("warfare_rules")
            if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
                skipped_missing_or_empty += 1
            elif isinstance(raw_value, dict) and not str(raw_value.get("warfare_rules", "")).strip():
                skipped_missing_or_empty += 1
            else:
                skipped_invalid += 1
            continue
        kept.append(match)
        per_rule_match_count[rule] += 1

    stats = {
        "raw_match_count": len(raw_matches),
        "valid_rule_match_count": len(kept),
        "skipped_missing_or_empty_warfare_rules_count": skipped_missing_or_empty,
        "skipped_invalid_warfare_rules_count": skipped_invalid,
        "per_rule_match_count": per_rule_match_count,
    }
    return kept, stats


def load_valid_matches(raw_path: Path = RAW_JSONL_PATH) -> list[dict[str, Any]]:
    raw_path = Path(raw_path)
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path

    matches: list[dict[str, Any]] = []
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = json.loads(line)
            if validate_match_record(match):
                continue
            matches.append(match)
    return matches


def split_match_ids(
    match_ids: list[int],
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[set[int], set[int]]:
    """Assign whole matches to train or validation (never both)."""
    unique_ids = np.array(sorted(set(match_ids)))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(unique_ids)
    val_count = max(1, int(round(len(perm) * val_ratio)))
    val_ids = set(int(x) for x in perm[:val_count])
    train_ids = set(int(x) for x in perm[val_count:])
    if not train_ids:
        train_ids = val_ids.copy()
        val_ids = set(int(x) for x in perm[:val_count])
    return train_ids, val_ids


def side_order_for_first_pick(first_pick_side: str) -> np.ndarray:
    if first_pick_side == "ally":
        return SIDE_ORDER_ALLY_FIRST
    return SIDE_ORDER_ENEMY_FIRST


def first_pick_side_id(first_pick_side: str) -> int:
    return 0 if first_pick_side == "ally" else 1


def side_id_from_name(side: str) -> int:
    return ALLY_SIDE_ID if side == "ally" else ENEMY_SIDE_ID


def build_candidate_mask(
    encoders: DraftEncoders,
    *,
    draft_hero_ids: np.ndarray,
    ally_preban_ids: np.ndarray,
    enemy_preban_ids: np.ndarray,
) -> np.ndarray:
    mask = np.ones(encoders.num_heroes, dtype=np.float32)
    unavailable = set(draft_hero_ids.tolist())
    unavailable.update(ally_preban_ids.tolist())
    unavailable.update(enemy_preban_ids.tolist())
    unavailable.discard(PAD_HERO_ID)
    for hero_id in unavailable:
        if 0 <= hero_id < encoders.num_heroes:
            mask[hero_id] = 0.0
    mask[PAD_HERO_ID] = 0.0
    mask[UNK_HERO_ID] = 1.0
    return mask


def samples_from_match(match: dict[str, Any], encoders: DraftEncoders) -> list[DraftSample]:
    draft = sorted(match["draft"], key=lambda x: x["order"])
    if len(draft) != DRAFT_SLOTS:
        return []

    warfare_rule = extract_concrete_warfare_rule(match)
    if warfare_rule is None:
        return []
    encoders._require_warfare_rule_encoders()
    warfare_rule_id = encoders.warfare_rule_id_for_name(warfare_rule)

    fps = match["first_pick_side"]
    side_order = side_order_for_first_pick(fps)
    is_first_pick_slot = (side_order == (ALLY_SIDE_ID if fps == "ally" else ENEMY_SIDE_ID)).astype(np.int32)

    ally_preban = [encoders.hero_id(h) for h in (match.get("ally_preban") or [])[:MAX_PREBAN]]
    enemy_preban = [encoders.hero_id(h) for h in (match.get("enemy_preban") or [])[:MAX_PREBAN]]
    ally_preban_ids = np.array(ally_preban + [PAD_HERO_ID] * (MAX_PREBAN - len(ally_preban)), dtype=np.int32)
    enemy_preban_ids = np.array(enemy_preban + [PAD_HERO_ID] * (MAX_PREBAN - len(enemy_preban)), dtype=np.int32)
    ally_preban_role_ids = encoders.role_ids_for_hero_ids(ally_preban_ids)
    ally_preban_element_ids = encoders.element_ids_for_hero_ids(ally_preban_ids)
    enemy_preban_role_ids = encoders.role_ids_for_hero_ids(enemy_preban_ids)
    enemy_preban_element_ids = encoders.element_ids_for_hero_ids(enemy_preban_ids)

    order_to_hero: dict[int, str] = {entry["order"]: entry["hero"] for entry in draft}
    order_to_side: dict[int, str] = {entry["order"]: entry["side"] for entry in draft}

    samples: list[DraftSample] = []
    draft_order_ids = np.arange(1, DRAFT_SLOTS + 1, dtype=np.int32)
    draft_position_bucket_ids = encoders.draft_position_bucket_ids()

    for step in range(1, DRAFT_SLOTS):
        hero_ids = np.zeros(DRAFT_SLOTS, dtype=np.int32)
        side_ids = np.zeros(DRAFT_SLOTS, dtype=np.int32)
        for order in range(1, step + 1):
            slot_idx = order - 1
            hero_ids[slot_idx] = encoders.hero_id(order_to_hero[order])
            side_ids[slot_idx] = side_id_from_name(order_to_side[order])
        role_ids = encoders.role_ids_for_hero_ids(hero_ids)
        element_ids = encoders.element_ids_for_hero_ids(hero_ids)

        next_order = step + 1
        target_hero = encoders.hero_id(order_to_hero[next_order])
        target_side = side_id_from_name(order_to_side[next_order])
        current_position_bucket_id = encoders.position_bucket_id_for_order(next_order)
        candidate_mask = build_candidate_mask(
            encoders,
            draft_hero_ids=hero_ids,
            ally_preban_ids=ally_preban_ids,
            enemy_preban_ids=enemy_preban_ids,
        )

        samples.append(
            DraftSample(
                draft_hero_ids=hero_ids,
                draft_role_ids=role_ids,
                draft_element_ids=element_ids,
                draft_side_ids=side_ids,
                draft_order_ids=draft_order_ids.copy(),
                draft_position_bucket_ids=draft_position_bucket_ids.copy(),
                is_first_pick_slot=is_first_pick_slot.copy(),
                ally_preban_ids=ally_preban_ids.copy(),
                ally_preban_role_ids=ally_preban_role_ids.copy(),
                ally_preban_element_ids=ally_preban_element_ids.copy(),
                enemy_preban_ids=enemy_preban_ids.copy(),
                enemy_preban_role_ids=enemy_preban_role_ids.copy(),
                enemy_preban_element_ids=enemy_preban_element_ids.copy(),
                candidate_mask=candidate_mask,
                first_pick_side=first_pick_side_id(fps),
                draft_step=step,
                match_id=int(match["match_id"]),
                target_next_hero=target_hero,
                target_next_side=target_side,
                current_position_bucket_id=current_position_bucket_id,
                current_order=next_order,
                current_side_id=target_side,
                warfare_rule_id=warfare_rule_id,
            )
        )

    return samples


def build_datasets(
    matches: list[dict[str, Any]],
    encoders: DraftEncoders,
    train_match_ids: set[int],
    val_match_ids: set[int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    train_samples: list[DraftSample] = []
    val_samples: list[DraftSample] = []

    for match in matches:
        mid = int(match["match_id"])
        samples = samples_from_match(match, encoders)
        if mid in train_match_ids:
            train_samples.extend(samples)
        elif mid in val_match_ids:
            val_samples.extend(samples)

    return samples_to_arrays(train_samples), samples_to_arrays(val_samples)


def samples_to_arrays(samples: list[DraftSample]) -> dict[str, np.ndarray]:
    if not samples:
        return {}

    def stack(attr: str, dtype) -> np.ndarray:
        return np.stack([getattr(s, attr) for s in samples]).astype(dtype)

    return {
        "draft_hero_ids": stack("draft_hero_ids", np.int32),
        "draft_role_ids": stack("draft_role_ids", np.int32),
        "draft_element_ids": stack("draft_element_ids", np.int32),
        "draft_side_ids": stack("draft_side_ids", np.int32),
        "draft_order_ids": stack("draft_order_ids", np.int32),
        "draft_position_bucket_ids": stack("draft_position_bucket_ids", np.int32),
        "is_first_pick_slot": stack("is_first_pick_slot", np.int32),
        "ally_preban_ids": stack("ally_preban_ids", np.int32),
        "ally_preban_role_ids": stack("ally_preban_role_ids", np.int32),
        "ally_preban_element_ids": stack("ally_preban_element_ids", np.int32),
        "enemy_preban_ids": stack("enemy_preban_ids", np.int32),
        "enemy_preban_role_ids": stack("enemy_preban_role_ids", np.int32),
        "enemy_preban_element_ids": stack("enemy_preban_element_ids", np.int32),
        "candidate_mask": stack("candidate_mask", np.float32),
        "current_position_bucket_id": stack("current_position_bucket_id", np.int32),
        "current_order": stack("current_order", np.int32),
        "current_side_id": stack("current_side_id", np.int32),
        "warfare_rule_id": stack("warfare_rule_id", np.int32),
        "first_pick_side": stack("first_pick_side", np.int32),
        "draft_step": stack("draft_step", np.int32),
        "match_id": stack("match_id", np.int32),
        "target_next_hero": stack("target_next_hero", np.int32),
        "target_next_side": stack("target_next_side", np.int32),
    }


def compute_warfare_rule_sample_counts(arrays: dict[str, np.ndarray]) -> dict[str, int]:
    counts = {rule: 0 for rule in WARFARE_RULE_ORDER}
    if "warfare_rule_id" not in arrays:
        return counts
    for rule_id in arrays["warfare_rule_id"].astype(np.int64):
        if 1 <= int(rule_id) <= len(WARFARE_RULE_ORDER):
            counts[WARFARE_RULE_ORDER[int(rule_id) - 1]] += 1
    return counts


def compute_warfare_rule_priors(sample_counts: dict[str, int]) -> dict[str, float]:
    total = float(sum(sample_counts.get(rule, 0) for rule in WARFARE_RULE_ORDER))
    if total <= 0:
        uniform = 1.0 / len(WARFARE_RULE_ORDER)
        return {rule: uniform for rule in WARFARE_RULE_ORDER}
    return {rule: sample_counts.get(rule, 0) / total for rule in WARFARE_RULE_ORDER}


def normalize_warfare_rules_param(raw_value: str | None) -> str:
    text = (raw_value or "").strip()
    if not text or text.upper() == WARFARE_RULE_ANY:
        return WARFARE_RULE_ANY
    for rule in WARFARE_RULE_ORDER:
        if text == rule:
            return rule
    raise ValueError(
        f"Invalid warfare_rules value {raw_value!r}; expected ANY or one of {WARFARE_RULE_ORDER}"
    )


def clone_inference_arrays_with_rule(
    base_arrays: dict[str, np.ndarray],
    encoders: DraftEncoders,
    warfare_rule: str,
) -> dict[str, np.ndarray]:
    cloned = {key: value.copy() for key, value in base_arrays.items()}
    rule_id = encoders.warfare_rule_id_for_name(warfare_rule)
    cloned["warfare_rule_id"] = np.array([rule_id], dtype=np.int32)
    return cloned


def mask_and_normalize_probs(
    probs: np.ndarray,
    candidate_mask: np.ndarray,
) -> np.ndarray:
    masked = probs.astype(np.float64).copy()
    masked[candidate_mask <= 0] = 0.0
    total = float(masked.sum())
    if total <= 0.0:
        return masked
    return masked / total


def blend_concrete_rule_probabilities(
    probs_by_rule: dict[str, np.ndarray],
    priors: dict[str, float],
) -> np.ndarray:
    if not probs_by_rule:
        raise ValueError("Cannot blend warfare rule probabilities without concrete rule outputs")
    num_heroes = next(iter(probs_by_rule.values())).shape[0]
    blended = np.zeros(num_heroes, dtype=np.float64)
    weight_sum = 0.0
    for rule in WARFARE_RULE_ORDER:
        if rule not in probs_by_rule:
            continue
        weight = float(priors.get(rule, 0.0))
        if weight <= 0.0:
            continue
        blended += weight * probs_by_rule[rule].astype(np.float64)
        weight_sum += weight
    if weight_sum <= 0.0:
        for rule in WARFARE_RULE_ORDER:
            blended += probs_by_rule[rule].astype(np.float64)
        return blended / len(WARFARE_RULE_ORDER)
    return blended / weight_sum


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


FIRST_PICK_SLOT_INDICES = [0, 3, 4, 7, 8]


def trim_draft_picks(first_team_picks: list[str], non_first_team_picks: list[str]) -> tuple[list[str], list[str]]:
    """Mirror recommender_service.process_picks for inference-time draft reconstruction."""
    max_non_first_team_picks = [0, 2, 2, 4, 4, 5]
    max_first_team_picks = [1, 1, 3, 3, 5, 5, 5]

    first_team_picks = list(first_team_picks[:5])
    non_first_team_picks = list(non_first_team_picks[:5])

    if len(first_team_picks) < 6:
        non_first_team_picks = non_first_team_picks[: max_non_first_team_picks[len(first_team_picks)]]
    if len(non_first_team_picks) < 6:
        first_team_picks = first_team_picks[: max_first_team_picks[len(non_first_team_picks)]]

    return first_team_picks[:5], non_first_team_picks[:5]


def interleave_api_picks(
    user_picks: list[str],
    enemy_picks: list[str],
    first_pick_team: str,
) -> tuple[list[tuple[str, str]], str]:
    """Return draft-order (hero_code, side) pairs and first-pick side name."""
    if first_pick_team == "My Team":
        user_picks, enemy_picks = trim_draft_picks(user_picks, enemy_picks)
        first_pick_side = "ally"
    else:
        enemy_picks, user_picks = trim_draft_picks(enemy_picks, user_picks)
        first_pick_side = "enemy"

    ordered: list[tuple[str, str]] = []
    user_idx = 0
    enemy_idx = 0
    for slot in range(len(user_picks) + len(enemy_picks)):
        if first_pick_side == "ally":
            if slot in FIRST_PICK_SLOT_INDICES:
                hero = user_picks[user_idx]
                user_idx += 1
                side = "ally"
            else:
                hero = enemy_picks[enemy_idx]
                enemy_idx += 1
                side = "enemy"
        else:
            if slot in FIRST_PICK_SLOT_INDICES:
                hero = enemy_picks[enemy_idx]
                enemy_idx += 1
                side = "enemy"
            else:
                hero = user_picks[user_idx]
                user_idx += 1
                side = "ally"
        ordered.append((hero, side))
    return ordered, first_pick_side


def build_inference_arrays(
    encoders: DraftEncoders,
    *,
    user_picks: list[str],
    enemy_picks: list[str],
    first_pick_team: str,
    ally_preban: list[str] | None = None,
    enemy_preban: list[str] | None = None,
    warfare_rules: str = WARFARE_RULE_ANY,
) -> dict[str, np.ndarray]:
    """Build a single-sample batch of Transformer inputs from API draft state."""
    encoders._require_warfare_rule_encoders()
    normalized_rule = normalize_warfare_rules_param(warfare_rules)
    if normalized_rule != WARFARE_RULE_ANY:
        warfare_rule_id = encoders.warfare_rule_id_for_name(normalized_rule)
    else:
        warfare_rule_id = encoders.warfare_rule_id_for_name(WARFARE_RULE_ORDER[0])

    ordered, first_pick_side = interleave_api_picks(user_picks, enemy_picks, first_pick_team)
    side_order = side_order_for_first_pick(first_pick_side)
    first_pick_side_token = ALLY_SIDE_ID if first_pick_side == "ally" else ENEMY_SIDE_ID
    is_first_pick_slot = (side_order == first_pick_side_token).astype(np.int32)
    draft_order_ids = np.arange(1, DRAFT_SLOTS + 1, dtype=np.int32)

    hero_ids = np.zeros(DRAFT_SLOTS, dtype=np.int32)
    side_ids = np.zeros(DRAFT_SLOTS, dtype=np.int32)
    for slot_idx, (hero_code, side_name) in enumerate(ordered):
        hero_ids[slot_idx] = encoders.hero_id(None if hero_code == "unknown" else hero_code)
        side_ids[slot_idx] = side_id_from_name(side_name)
    role_ids = encoders.role_ids_for_hero_ids(hero_ids)
    element_ids = encoders.element_ids_for_hero_ids(hero_ids)

    ally_codes = (ally_preban or [])[:MAX_PREBAN]
    enemy_codes = (enemy_preban or [])[:MAX_PREBAN]
    ally_preban_ids = np.array(
        [encoders.hero_id(code) for code in ally_codes] + [PAD_HERO_ID] * (MAX_PREBAN - len(ally_codes)),
        dtype=np.int32,
    )
    enemy_preban_ids = np.array(
        [encoders.hero_id(code) for code in enemy_codes] + [PAD_HERO_ID] * (MAX_PREBAN - len(enemy_codes)),
        dtype=np.int32,
    )
    ally_preban_role_ids = encoders.role_ids_for_hero_ids(ally_preban_ids)
    ally_preban_element_ids = encoders.element_ids_for_hero_ids(ally_preban_ids)
    enemy_preban_role_ids = encoders.role_ids_for_hero_ids(enemy_preban_ids)
    enemy_preban_element_ids = encoders.element_ids_for_hero_ids(enemy_preban_ids)
    candidate_mask = build_candidate_mask(
        encoders,
        draft_hero_ids=hero_ids,
        ally_preban_ids=ally_preban_ids,
        enemy_preban_ids=enemy_preban_ids,
    )

    draft_step = len(ordered)
    next_order = min(draft_step + 1, DRAFT_SLOTS)
    current_side_id = int(side_order[draft_step]) if draft_step < DRAFT_SLOTS else int(side_order[-1])
    current_position_bucket_id = encoders.position_bucket_id_for_order(next_order)

    return {
        "draft_hero_ids": hero_ids[np.newaxis, :],
        "draft_role_ids": role_ids[np.newaxis, :],
        "draft_element_ids": element_ids[np.newaxis, :],
        "draft_side_ids": side_ids[np.newaxis, :],
        "draft_order_ids": draft_order_ids[np.newaxis, :],
        "draft_position_bucket_ids": encoders.draft_position_bucket_ids()[np.newaxis, :],
        "is_first_pick_slot": is_first_pick_slot[np.newaxis, :],
        "ally_preban_ids": ally_preban_ids[np.newaxis, :],
        "ally_preban_role_ids": ally_preban_role_ids[np.newaxis, :],
        "ally_preban_element_ids": ally_preban_element_ids[np.newaxis, :],
        "enemy_preban_ids": enemy_preban_ids[np.newaxis, :],
        "enemy_preban_role_ids": enemy_preban_role_ids[np.newaxis, :],
        "enemy_preban_element_ids": enemy_preban_element_ids[np.newaxis, :],
        "candidate_mask": candidate_mask[np.newaxis, :],
        "current_position_bucket_id": np.array([current_position_bucket_id], dtype=np.int32),
        "current_order": np.array([next_order], dtype=np.int32),
        "current_side_id": np.array([current_side_id], dtype=np.int32),
        "warfare_rule_id": np.array([warfare_rule_id], dtype=np.int32),
        "first_pick_side": np.array([first_pick_side_id(first_pick_side)], dtype=np.int32),
        "draft_step": np.array([draft_step], dtype=np.int32),
    }
