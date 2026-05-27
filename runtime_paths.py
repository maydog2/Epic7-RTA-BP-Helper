"""Resolve bundled and writable paths for dev checkout and PyInstaller builds."""

from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def get_bundle_dir() -> Path:
    """Directory containing read-only bundled resources."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parent


def get_app_base_dir() -> Path:
    """Directory for writable runtime files such as logs."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resource_path(*parts: str) -> Path:
    return get_bundle_dir().joinpath(*parts)


APP_BASE_DIR = get_app_base_dir()
BUNDLE_DIR = get_bundle_dir()
DATA_DIR = resource_path("data")
TRANSFORMER_DATA_DIR = DATA_DIR / "transformer"
RERANKER_DATA_DIR = DATA_DIR / "reranker"
RUNTIME_DATA_DIR = DATA_DIR / "runtime"
HERO_DETAILS_PATH = DATA_DIR / "hero_details.csv"
RAW_MATCH_HISTORY_PATH = DATA_DIR / "epic7_match_history_raw.jsonl"
PREBAN_STATS_PATH = RUNTIME_DATA_DIR / "preban_stats.pkl"
FIRST_PICK_RECORDS_PATH = RUNTIME_DATA_DIR / "first_pick_records.pkl"
PORTRAIT_DIR = resource_path("dataset")
ELEMENT_ICON_DIR = resource_path("CharacterUI", "elements")
ROLE_ICON_DIR = resource_path("CharacterUI", "roles")
FRONTEND_DIST_DIR = resource_path("frontend", "dist")
WORKFLOW_DIR = resource_path("workflow_scripts")
