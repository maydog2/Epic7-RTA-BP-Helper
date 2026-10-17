"""Export Phase 1 Android assets: hero JSON, portraits, icons, and frontend dist."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANDROID_ASSETS = PROJECT_ROOT / "E7_BP_Helper_Android" / "app" / "src" / "main" / "assets"
HERO_DETAILS_PATH = PROJECT_ROOT / "data" / "hero_details.csv"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
PORTRAIT_DIR = PROJECT_ROOT / "dataset"
ELEMENT_ICON_DIR = PROJECT_ROOT / "CharacterUI" / "elements"
ROLE_ICON_DIR = PROJECT_ROOT / "CharacterUI" / "roles"

HERO_DETAIL_ALIASES = {
    "c0002": "c1005",  # Mercedes
}


def parse_non_negative_int(raw_value: str | None) -> int:
    if raw_value is None:
        return 0
    try:
        return max(int(float(raw_value)), 0)
    except (TypeError, ValueError):
        return 0


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


def build_hero_options(rows_by_code: dict[str, dict[str, str]]) -> list[dict[str, object]]:
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
                "avatar_url": f"/dataset/{code}/c.png"
                if (PORTRAIT_DIR / code / "c.png").exists()
                else "",
                "element_icon_url": f"/CharacterUI/elements/{element}.png"
                if element and (ELEMENT_ICON_DIR / f"{element}.png").exists()
                else "",
                "role_icon_url": f"/CharacterUI/roles/{role}.png"
                if role and (ROLE_ICON_DIR / f"{role}.png").exists()
                else "",
            }
        )
    return heroes


def load_rows_by_code(path: Path) -> dict[str, dict[str, str]]:
    rows_by_code: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            code = (row.get("Hero") or "").strip()
            if code:
                rows_by_code[code] = row
    return rows_by_code


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_portraits(heroes: list[dict[str, object]], dst_root: Path) -> int:
    copied = 0
    for hero in heroes:
        code = str(hero["code"])
        src = PORTRAIT_DIR / code / "c.png"
        if not src.is_file():
            continue
        dst = dst_root / code
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst / "c.png")
        copied += 1
    return copied


def export_assets(
    *,
    output_dir: Path,
    skip_frontend: bool = False,
) -> dict[str, int]:
    if not HERO_DETAILS_PATH.is_file():
        raise FileNotFoundError(f"Missing hero details CSV: {HERO_DETAILS_PATH}")

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows_by_code = load_rows_by_code(HERO_DETAILS_PATH)
    heroes = build_hero_options(rows_by_code)
    hero_details_path = data_dir / "hero_details.json"
    hero_details_path.write_text(
        json.dumps({"heroes": heroes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    portrait_count = copy_portraits(heroes, output_dir / "dataset")
    copy_tree(ELEMENT_ICON_DIR, output_dir / "CharacterUI" / "elements")
    copy_tree(ROLE_ICON_DIR, output_dir / "CharacterUI" / "roles")

    frontend_copied = 0
    if not skip_frontend:
        if not (FRONTEND_DIST / "index.html").is_file():
            raise FileNotFoundError(
                f"Missing frontend build at {FRONTEND_DIST}. Run `npm run build` in frontend/ first."
            )
        copy_tree(FRONTEND_DIST, output_dir / "public")
        frontend_copied = len(list((output_dir / "public").rglob("*")))

    return {
        "heroes": len(heroes),
        "portraits": portrait_count,
        "frontend_files": frontend_copied,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Phase 1 Android offline assets.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ANDROID_ASSETS,
        help="Android assets output directory.",
    )
    parser.add_argument(
        "--skip-frontend",
        action="store_true",
        help="Only export hero data/images; do not copy frontend/dist.",
    )
    args = parser.parse_args()

    try:
        stats = export_assets(output_dir=args.output, skip_frontend=args.skip_frontend)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Exported Android Phase 1 assets to {args.output}")
    print(f"  heroes: {stats['heroes']}")
    print(f"  portraits: {stats['portraits']}")
    if not args.skip_frontend:
        print(f"  frontend files: {stats['frontend_files']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
