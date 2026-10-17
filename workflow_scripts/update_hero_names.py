"""Update English and simplified Chinese names from Stove herorecord.

Supports targeted hero codes or a full refresh via --all.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from zhconv import convert as zh_convert

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hero_details_io import (  # noqa: E402
    DEFAULT_HERO_DETAILS_PATH,
    load_hero_details_by_code,
    save_hero_detail_rows,
    sorted_hero_detail_rows,
    upsert_hero_detail_row,
)

EN_HERORECORD_URL = "https://epic7.onstove.com/en/gg/herorecord"
ZH_TW_HERORECORD_URL = "https://epic7.onstove.com/zh-TW/gg/herorecord"
HERO_LIST_SELECTOR = "div.filterlist-wrap ul li"
HERO_CODE_PATTERN = re.compile(r"^c\d{4}$", re.I)
_TRAD_TO_SIMP = "zh-cn"


def normalize_hero_codes(raw_codes: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in raw_codes:
        for part in re.split(r"[\s,]+", raw.strip()):
            if not part:
                continue
            code = part.strip().lower()
            if not HERO_CODE_PATTERN.fullmatch(code):
                raise ValueError(f"Invalid hero code: {part!r} (expected format c1234)")
            if code in seen:
                continue
            seen.add(code)
            normalized.append(code)
    return normalized


def traditional_to_simplified(text: str) -> str:
    return zh_convert(text.strip(), _TRAD_TO_SIMP)


def create_driver() -> webdriver.Chrome:
    chrome_options = Options()
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--headless")
    service = ChromeService(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)


def wait_for_hero_list(driver: webdriver.Chrome, url: str) -> list[object]:
    driver.get(url)
    WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, HERO_LIST_SELECTOR)))
    time.sleep(8)
    return WebDriverWait(driver, 15).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, HERO_LIST_SELECTOR))
    )


def scrape_names_from_page(
    driver: webdriver.Chrome,
    *,
    url: str,
    convert_zh: bool,
    wanted: set[str] | None = None,
    hero_limit: int = 0,
) -> tuple[dict[str, str], dict[str, str]]:
    names: dict[str, str] = {}
    portrait_urls: dict[str, str] = {}
    remaining = set(wanted) if wanted is not None else None
    hero_elements = wait_for_hero_list(driver, url)
    label = "name_zh" if convert_zh else "name"

    for index, hero_element in enumerate(hero_elements):
        if remaining is not None and not remaining:
            break
        if remaining is None and hero_limit > 0 and index >= hero_limit:
            print(f"Stopping at hero_limit={hero_limit}")
            break

        try:
            img_element = hero_element.find_element(By.TAG_NAME, "img")
            hero_code = (img_element.get_attribute("alt") or "").strip().lower()
            if remaining is not None:
                if hero_code not in remaining:
                    continue
            elif not HERO_CODE_PATTERN.fullmatch(hero_code):
                continue

            display_name = hero_element.find_element(By.TAG_NAME, "i").text.strip()
            image_url = (img_element.get_attribute("src") or "").strip()
            if not display_name:
                continue

            names[hero_code] = traditional_to_simplified(display_name) if convert_zh else display_name
            if image_url:
                portrait_urls[hero_code] = image_url
            if remaining is not None:
                remaining.remove(hero_code)
            print(f"Found {hero_code} {label}={names[hero_code]!r}")
        except Exception as exc:
            print(f"Error reading hero list entry: {exc}")

    return names, portrait_urls


def scrape_all_hero_names(
    *,
    driver: webdriver.Chrome,
    hero_limit: int = 0,
    zh_only: bool = False,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Fetch names for every hero on Stove herorecord."""
    english_names: dict[str, str] = {}
    portrait_urls: dict[str, str] = {}

    if not zh_only:
        english_names, en_portraits = scrape_names_from_page(
            driver,
            url=EN_HERORECORD_URL,
            convert_zh=False,
            hero_limit=hero_limit,
        )
        portrait_urls.update(en_portraits)

    chinese_names, zh_portraits = scrape_names_from_page(
        driver,
        url=ZH_TW_HERORECORD_URL,
        convert_zh=True,
        hero_limit=hero_limit,
    )
    portrait_urls.update({code: url for code, url in zh_portraits.items() if code not in portrait_urls})
    return english_names, chinese_names, portrait_urls


def scrape_hero_names_for_codes(
    hero_codes: list[str],
    *,
    driver: webdriver.Chrome,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Fetch English, simplified Chinese, and portrait URLs for specific hero codes."""
    normalized_codes = normalize_hero_codes(hero_codes)
    if not normalized_codes:
        return {}, {}, {}

    english_names, portrait_urls = scrape_names_from_page(
        driver,
        url=EN_HERORECORD_URL,
        convert_zh=False,
        wanted=set(normalized_codes),
    )
    chinese_names, zh_portrait_urls = scrape_names_from_page(
        driver,
        url=ZH_TW_HERORECORD_URL,
        convert_zh=True,
        wanted=set(normalized_codes),
    )
    portrait_urls.update({code: url for code, url in zh_portrait_urls.items() if code not in portrait_urls})
    return english_names, chinese_names, portrait_urls


def merge_name_updates_into_rows(
    rows: list[dict[str, str]],
    hero_codes: list[str],
    *,
    english_names: dict[str, str],
    chinese_names: dict[str, str],
) -> list[str]:
    """Apply scraped names to in-memory hero detail rows. Returns updated hero codes."""
    updated_codes: list[str] = []
    for code in normalize_hero_codes(hero_codes):
        update: dict[str, str] = {"Hero": code}
        if code in english_names:
            update["name"] = english_names[code]
        if code in chinese_names:
            update["name_zh"] = chinese_names[code]
        if update.keys() == {"Hero"}:
            continue
        upsert_hero_detail_row(rows, update)
        updated_codes.append(code)
    return updated_codes


def _save_portraits_for_codes(
    hero_codes: list[str],
    portrait_urls: dict[str, str],
) -> None:
    for code in hero_codes:
        image_url = portrait_urls.get(code, "")
        if image_url:
            save_hero_portrait(code, image_url)
        else:
            print(f"No portrait URL found for {code}")


def update_hero_names_for_codes(
    hero_codes: list[str],
    *,
    rows: list[dict[str, str]],
    driver: webdriver.Chrome,
    save_portraits: bool = False,
) -> list[str]:
    """Scrape Stove names for hero codes and merge them into the provided rows."""
    normalized_codes = normalize_hero_codes(hero_codes)
    if not normalized_codes:
        return []

    english_names, chinese_names, portrait_urls = scrape_hero_names_for_codes(
        normalized_codes,
        driver=driver,
    )
    updated_codes = merge_name_updates_into_rows(
        rows,
        normalized_codes,
        english_names=english_names,
        chinese_names=chinese_names,
    )
    if updated_codes:
        print(f"Updated names for: {', '.join(updated_codes)}")

    if save_portraits:
        _save_portraits_for_codes(normalized_codes, portrait_urls)

    return [
        code
        for code in normalized_codes
        if code not in english_names and code not in chinese_names
    ]


def update_all_hero_names(
    *,
    rows: list[dict[str, str]],
    driver: webdriver.Chrome,
    hero_limit: int = 0,
    zh_only: bool = False,
    save_portraits: bool = False,
) -> tuple[int, list[str]]:
    """Scrape all Stove herorecord names and merge them into the provided rows."""
    existing_codes = {row.get("Hero") for row in rows if row.get("Hero")}
    english_names, chinese_names, portrait_urls = scrape_all_hero_names(
        driver=driver,
        hero_limit=hero_limit,
        zh_only=zh_only,
    )
    if not english_names and not chinese_names:
        return 0, []

    all_codes = sorted(set(english_names) | set(chinese_names))
    updated_codes = merge_name_updates_into_rows(
        rows,
        all_codes,
        english_names=english_names,
        chinese_names=chinese_names,
    )
    missing_in_csv = [code for code in all_codes if code not in existing_codes]

    print(f"Updated {len(updated_codes)} hero names in hero_details.csv")
    if missing_in_csv:
        print(f"Codes scraped but absent from previous CSV: {len(missing_in_csv)}")

    codes_without_zh = [row["Hero"] for row in rows if row.get("Hero") and not row.get("name_zh")]
    if codes_without_zh:
        print(f"Rows still missing name_zh: {len(codes_without_zh)}")

    if save_portraits:
        _save_portraits_for_codes(all_codes, portrait_urls)

    return len(updated_codes), missing_in_csv


def save_hero_portrait(hero_code: str, image_url: str) -> bool:
    hero_dir = PROJECT_ROOT / "dataset" / hero_code
    hero_dir.mkdir(parents=True, exist_ok=True)
    image_path = hero_dir / "c.png"

    try:
        response = requests.get(image_url, timeout=30)
        content_type = response.headers.get("Content-Type", "")
        if response.status_code == 200 and response.content and "image" in content_type:
            image_path.write_bytes(response.content)
            print(f"Saved portrait: {image_path}")
            return True
        print(f"Failed to download portrait for {hero_code}: {response.status_code} {content_type}")
    except requests.RequestException as exc:
        print(f"Failed to download portrait for {hero_code}: {exc}")
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update English and simplified Chinese names in hero_details.csv from Stove.",
    )
    parser.add_argument(
        "heroes",
        nargs="*",
        help="Hero codes to update, e.g. c2185 c2186",
    )
    parser.add_argument(
        "--heroes",
        dest="heroes_csv",
        default="",
        help="Comma-separated hero codes, e.g. c2185,c2186",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Refresh every hero on Stove herorecord.",
    )
    parser.add_argument(
        "--zh-only",
        action="store_true",
        help="With --all, only refresh name_zh.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.environ.get("E7_CHARACTER_IDS_LIMIT", "0") or "0"),
        help="With --all, optional smoke-test limit for number of heroes scraped.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_HERO_DETAILS_PATH,
        help="CSV path to update (default: data/hero_details.csv).",
    )
    parser.add_argument(
        "--download-portrait",
        action="store_true",
        help="Also download dataset/<hero_code>/c.png from Stove.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hero_codes = normalize_hero_codes(args.heroes + ([args.heroes_csv] if args.heroes_csv else []))
    if args.all and hero_codes:
        print("Ignoring positional hero codes because --all was provided.")
    if not args.all and not hero_codes:
        print(
            "No hero codes provided. Examples:\n"
            "  python workflow_scripts/update_hero_names.py c2185\n"
            "  python workflow_scripts/update_hero_names.py --all"
        )
        return 2

    existing_rows = load_hero_details_by_code(args.output)
    merged_rows = list(existing_rows.values())

    driver = create_driver()
    try:
        if args.all:
            updated_count, missing_in_csv = update_all_hero_names(
                rows=merged_rows,
                driver=driver,
                hero_limit=args.limit,
                zh_only=args.zh_only,
                save_portraits=args.download_portrait,
            )
            if updated_count == 0:
                print("No hero names scraped.")
                return 1
        else:
            missing = update_hero_names_for_codes(
                hero_codes,
                rows=merged_rows,
                driver=driver,
                save_portraits=args.download_portrait,
            )
            if missing:
                print(f"Not found on Stove herorecord: {', '.join(missing)}")
    finally:
        driver.quit()

    save_hero_detail_rows(sorted_hero_detail_rows(merged_rows), args.output)
    if args.all:
        print(f"Saved refreshed hero names to {args.output}")
    else:
        print(f"Updated hero_details.csv for: {', '.join(hero_codes)}")
        return 1 if missing else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
