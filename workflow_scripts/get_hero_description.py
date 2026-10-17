"""Scrape hero role/element from CeciliaBot into data/hero_details.csv."""

from __future__ import annotations

import os
import re
import sys
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from hero_details_io import (
    DEFAULT_HERO_DETAILS_PATH,
    load_hero_detail_rows,
    save_hero_detail_rows,
    upsert_hero_detail_row,
)
from update_hero_names import create_driver, update_hero_names_for_codes

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

HERO_DETAIL_LIMIT = int(os.environ.get("E7_HERO_DETAILS_LIMIT", "0") or "0")
_HERO_DETAILS_FRESH = os.environ.get("E7_HERO_DETAILS_FRESH", "").lower() in ("1", "true", "yes")
_AUTO_UPDATE_HERO_NAMES = os.environ.get("E7_AUTO_UPDATE_HERO_NAMES", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
last_char_name = os.environ.get("E7_LAST_HERO_DETAIL_NAME", "Requiemroar")
_stove_driver: webdriver.Chrome | None = None

chrome_options = Options()
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--headless")
initial_url = "https://ceciliabot.github.io/#/hero/"
hero_grid_selector = (By.CSS_SELECTOR, "img.hero-img")


def init_driver() -> webdriver.Chrome:
    svc = ChromeService(ChromeDriverManager().install())
    return webdriver.Chrome(service=svc, options=chrome_options)


driver = init_driver()
driver.get(initial_url)


def wait_hero_grid(timeout: float = 45.0) -> None:
    WebDriverWait(driver, timeout).until(EC.presence_of_element_located(hero_grid_selector))


def recover_browser() -> None:
    global driver
    for attempt in range(4):
        try:
            driver.refresh()
            wait_hero_grid()
            return
        except (TimeoutException, WebDriverException) as exc:
            print(f"Browser recovery attempt {attempt + 1} failed: {exc}")
            try:
                driver.get(initial_url)
                wait_hero_grid()
                return
            except (TimeoutException, WebDriverException):
                if attempt >= 3:
                    raise
                try:
                    driver.quit()
                except WebDriverException:
                    pass
                driver = init_driver()
                driver.get(initial_url)
                wait_hero_grid()
                return


wait_hero_grid()

data: list[dict[str, str]] = []
if not _HERO_DETAILS_FRESH and DEFAULT_HERO_DETAILS_PATH.exists():
    data = load_hero_detail_rows(DEFAULT_HERO_DETAILS_PATH)

visited_heroes: list[str] = [row["Hero"] for row in data if row.get("Hero")]
if visited_heroes:
    print(f"Resuming with {len(visited_heroes)} heroes already in {DEFAULT_HERO_DETAILS_PATH}")


def get_stove_driver() -> webdriver.Chrome:
    global _stove_driver
    if _stove_driver is None:
        _stove_driver = create_driver()
    return _stove_driver


def close_stove_driver() -> None:
    global _stove_driver
    if _stove_driver is None:
        return
    try:
        _stove_driver.quit()
    except WebDriverException:
        pass
    _stove_driver = None


def parse_role_element(soup: BeautifulSoup) -> dict[str, str]:
    role = ""
    element = ""
    role_pattern = re.compile(r"cm_icon_role_(\w+)\.png")
    element_pattern = re.compile(r"cm_icon_prom(\w+)\.png")
    for img in soup.find_all("img", class_="inline-icon-image"):
        src = img.get("src", "")
        role_match = role_pattern.search(src)
        element_match = element_pattern.search(src)
        if role_match:
            role = role_match.group(1)
        if element_match:
            element = element_match.group(1)
    if element == "wind":
        element = "earth"
    return {"Role": role, "Element": element}


def maybe_update_hero_names(hero_code: str) -> None:
    if not _AUTO_UPDATE_HERO_NAMES:
        return

    current_row = next((row for row in data if row.get("Hero") == hero_code), None)
    if current_row and current_row.get("name") and current_row.get("name_zh"):
        return

    try:
        missing = update_hero_names_for_codes(
            [hero_code],
            rows=data,
            driver=get_stove_driver(),
        )
        if missing:
            print(f"Name lookup on Stove missed {hero_code}")
    except Exception as exc:
        print(f"Failed to update names for {hero_code}: {exc}")


def scrape_hero_detail(soup: BeautifulSoup, hero_code: str) -> None:
    detail = parse_role_element(soup)
    print(f"Scraped {hero_code}: role={detail['Role']}, element={detail['Element']}")
    upsert_hero_detail_row(data, {"Hero": hero_code, **detail})
    maybe_update_hero_names(hero_code)
    driver.execute_script("window.history.go(-1)")
    recover_browser()


is_done = False
hero_elements: list[object] = []

while True:
    try:
        if is_done:
            break

        hero_elements = driver.find_elements(*hero_grid_selector)

        for hero_element in hero_elements:
            raw_img = hero_element.get_attribute("data-src") or hero_element.get_attribute("src") or ""
            match = re.search(r"c\d{4}", raw_img)
            if not match:
                continue
            hero_code = match.group()

            if hero_code in visited_heroes:
                continue

            print(f"Processing hero: {hero_code}")
            visited_heroes.append(hero_code)
            ActionChains(driver).move_to_element(hero_element).perform()
            button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable(hero_element))
            button.click()
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "e7-chat-bubble")))

            soup = BeautifulSoup(driver.page_source, "html.parser")
            scrape_hero_detail(soup, hero_code)

            if HERO_DETAIL_LIMIT > 0 and len(visited_heroes) >= HERO_DETAIL_LIMIT:
                is_done = True
                break

            hero_name_el = soup.find("div", class_="page-title-box flex flex-col")
            current_hero_name = ""
            if hero_name_el and hero_name_el.find("h2"):
                current_hero_name = hero_name_el.find("h2").text.strip()
            if current_hero_name.strip().lower() == last_char_name.strip().lower():
                is_done = True
                break

    except Exception as exc:
        print(f"Crawl loop recovery: {exc}")
        recover_browser()

    finally:
        save_hero_detail_rows(data, DEFAULT_HERO_DETAILS_PATH)

driver.quit()
close_stove_driver()
print(len(hero_elements))
