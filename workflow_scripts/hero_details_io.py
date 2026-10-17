"""Read/write helpers for data/hero_details.csv."""

from __future__ import annotations

import csv
import time
from pathlib import Path

from hero_details_fields import (
    DEFAULT_HERO_DETAIL_ROW,
    HERO_DETAILS_FIELDNAMES,
    hero_code_from_row,
    normalize_hero_detail_row,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HERO_DETAILS_PATH = PROJECT_ROOT / "data" / "hero_details.csv"


def _replace_with_retries(tmp_path: Path, target_path: Path, *, attempts: int = 5) -> None:
    for attempt in range(attempts):
        try:
            tmp_path.replace(target_path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.25 * (attempt + 1))


def load_hero_detail_rows(path: Path = DEFAULT_HERO_DETAILS_PATH) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [normalize_hero_detail_row(dict(row)) for row in csv.DictReader(handle)]


def load_hero_details_by_code(path: Path = DEFAULT_HERO_DETAILS_PATH) -> dict[str, dict[str, str]]:
    return {
        row["Hero"]: row
        for row in load_hero_detail_rows(path)
        if row.get("Hero")
    }


def save_hero_detail_rows(
    rows: list[dict[str, str]],
    path: Path = DEFAULT_HERO_DETAILS_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HERO_DETAILS_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_hero_detail_row(row))
    _replace_with_retries(tmp_path, path)


def upsert_hero_detail_row(
    rows: list[dict[str, str]],
    update: dict[str, str],
) -> None:
    hero = hero_code_from_row(update)
    if not hero:
        return
    patch = {
        field: (update.get(field) or "").strip()
        for field in HERO_DETAILS_FIELDNAMES
        if field in update
    }
    patch["Hero"] = hero
    for index, existing in enumerate(rows):
        if existing.get("Hero") == hero:
            rows[index] = normalize_hero_detail_row({**existing, **patch})
            return
    rows.append(normalize_hero_detail_row({**DEFAULT_HERO_DETAIL_ROW, **patch}))


def sorted_hero_detail_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: row.get("Hero", ""))
