"""Shared column names for data/hero_details.csv."""

from __future__ import annotations

HERO_DETAILS_FIELDNAMES = [
    "Hero",
    "name",
    "name_zh",
    "Role",
    "Element",
    "appearance_count",
]

DEFAULT_HERO_DETAIL_ROW = {
    "name": "",
    "name_zh": "",
    "Role": "",
    "Element": "",
    "appearance_count": "0",
}


def hero_code_from_row(row: dict[str, str]) -> str:
    return (row.get("Hero") or row.get("code") or "").strip()


def normalize_hero_detail_row(row: dict[str, str]) -> dict[str, str]:
    hero = hero_code_from_row(row)
    normalized = {field: (row.get(field) or "").strip() for field in HERO_DETAILS_FIELDNAMES}
    normalized["Hero"] = hero
    if not normalized["appearance_count"]:
        normalized["appearance_count"] = DEFAULT_HERO_DETAIL_ROW["appearance_count"]
    return normalized
