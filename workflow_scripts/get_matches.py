import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import InvalidSessionIdException, StaleElementReferenceException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from match_history_utils import (
    ALLY_FINAL_BAN_TARGET,
    ENEMY_FINAL_BAN_TARGET,
    RAW_JSONL_PATH,
    detect_first_pick_side_from_battle,
    extract_played_at,
    extract_season_patch,
    extract_team_draft,
    extract_warfare_rules,
    is_match_within_days,
    make_draft_entry,
    merge_final_ban_targets,
    should_merge_final_ban_targets,
    validate_match_record,
)

# Smoke test: set E7_SCRAPER_SMOKE=1 to fetch only a few matches (pipeline check on Windows/macOS/Linux).
_SCRAPER_SMOKE = os.environ.get("E7_SCRAPER_SMOKE", "").lower() in ("1", "true", "yes")
_default_per_player = "5" if _SCRAPER_SMOKE else "100"
MAX_MATCHES_PER_PLAYER = int(
    os.environ.get(
        "E7_GET_MATCHES_MAX_PER_PLAYER",
        os.environ.get("E7_GET_MATCHES_MAX_PER_HERO", _default_per_player),
    )
)
SMOKE_PLAYER_PAGE_LIMIT = int(
    os.environ.get(
        "E7_GET_MATCHES_MAX_PLAYERS",
        os.environ.get("E7_GET_MATCHES_MAX_HEROES", "2" if _SCRAPER_SMOKE else "0"),
    )
)
WORKER_COUNT = max(1, int(os.environ.get("E7_GET_MATCHES_WORKERS", "4")))
RANK_SERVER = os.environ.get("E7_GET_MATCHES_RANK_SERVER", "All Servers").strip() or "All Servers"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


RANK_TOP_LIMIT = int(os.environ.get("E7_GET_MATCHES_RANK_TOP_LIMIT", "0") or "0")
MAX_PLAYERS_PER_RUN = int(os.environ.get("E7_GET_MATCHES_MAX_PLAYERS_PER_RUN", "0") or "0")
REVISIT_VISITED_PLAYERS = _env_flag("E7_GET_MATCHES_REVISIT")


MAX_MATCH_DAYS_ENABLED = _env_flag("E7_GET_MATCHES_MAX_DAYS_ENABLED")
MAX_MATCH_DAYS: int | None = None
if MAX_MATCH_DAYS_ENABLED:
    raw_max_days = os.environ.get("E7_GET_MATCHES_MAX_DAYS", "").strip()
    if not raw_max_days:
        print(
            "E7_GET_MATCHES_MAX_DAYS_ENABLED is set, but E7_GET_MATCHES_MAX_DAYS is missing; "
            "date filter disabled."
        )
        MAX_MATCH_DAYS_ENABLED = False
    else:
        try:
            MAX_MATCH_DAYS = max(1, int(raw_max_days))
        except ValueError:
            print(
                f"Invalid E7_GET_MATCHES_MAX_DAYS value {raw_max_days!r}; date filter disabled."
            )
            MAX_MATCH_DAYS_ENABLED = False

FIRST_PICK_ORDERS = [1, 4, 5, 8, 9]
SECOND_PICK_ORDERS = [2, 3, 6, 7, 10]
INITIAL_URL = "https://epic7.onstove.com/en/gg/rank/server"
SCRAPER_STATE_PATH = Path("data/epic7_scraper_state.json")
RESTART_THRESHOLD = 10
RUN_DURATION = timedelta(hours=5)

CHROME_DRIVER_PATH: str | None = None
_chrome_driver_lock = threading.Lock()


def match_signature(match: dict) -> tuple:
    """Stable identity for de-duping reruns; excludes local match_id."""
    draft = tuple(
        (entry["order"], entry["side"], entry["hero"])
        for entry in sorted(match["draft"], key=lambda item: item["order"])
    )
    return (
        match.get("first_pick_side"),
        match.get("winner_side"),
        tuple(match.get("ally_preban", [])),
        tuple(match.get("enemy_preban", [])),
        draft,
        match.get("season"),
        match.get("patch"),
        match.get("warfare_rules"),
    )


def wlog(worker_id: int, message: str) -> None:
    print(f"[worker {worker_id}] {message}", flush=True)


def replace_with_retries(tmp_path: Path, target_path: Path, *, attempts: int = 5) -> None:
    """Windows can briefly lock files shown in the editor; retry atomic replace."""
    for attempt in range(attempts):
        try:
            tmp_path.replace(target_path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.25 * (attempt + 1))


class ScraperCoordinator:
    """Thread-safe shared scraper state."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.visited_players: set[str] = set()
        self.in_progress_players: set[str] = set()
        self.match_signatures: set[tuple] = set()
        self.signature_to_index: dict[tuple, int] = {}
        self.matches: list[dict] = []
        self.next_match_id = 0
        self.start_time = datetime.now()
        self.end_time = self.start_time + RUN_DURATION
        self.players_completed_this_run = 0
        self.attempted_players_this_run: set[str] = set()

    def load_existing_matches(self) -> None:
        path = Path(RAW_JSONL_PATH)
        if not path.exists():
            return

        max_match_id = -1
        loaded: list[dict] = []
        signatures: set[tuple] = set()
        signature_to_index: dict[tuple, int] = {}
        with path.open(encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    match = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"Skipping malformed JSONL row while resuming: {exc}")
                    continue
                loaded.append(match)
                signature = match_signature(match)
                signatures.add(signature)
                signature_to_index[signature] = len(loaded) - 1
                max_match_id = max(max_match_id, int(match.get("match_id", -1)))

        with self.lock:
            self.matches = loaded
            self.match_signatures = signatures
            self.signature_to_index = signature_to_index
            self.next_match_id = max_match_id + 1
        print(
            f"Loaded {len(loaded)} existing matches from {path}; "
            f"next match_id={self.next_match_id}"
        )

    def load_scraper_state(self) -> None:
        if not SCRAPER_STATE_PATH.exists():
            return
        try:
            state = json.loads(SCRAPER_STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"Skipping malformed scraper state: {exc}")
            return

        if "visited_players" in state:
            visited = state.get("visited_players", [])
        else:
            visited = state.get("visited_heroes", [])

        with self.lock:
            self.visited_players.update(visited)
            self.in_progress_players.clear()
        print(
            f"Loaded {len(visited)} visited players from {SCRAPER_STATE_PATH}; "
            "in_progress_players starts empty"
        )

    def should_stop(self) -> bool:
        return self.stop.is_set() or datetime.now() >= self.end_time

    def try_claim_player(self, player_name: str) -> bool:
        with self.lock:
            if self.should_stop():
                return False
            if player_name in self.in_progress_players:
                return False
            if player_name in self.attempted_players_this_run:
                return False
            if not REVISIT_VISITED_PLAYERS and player_name in self.visited_players:
                return False
            self.attempted_players_this_run.add(player_name)
            self.in_progress_players.add(player_name)
            return True

    def complete_player_success(self, player_name: str) -> None:
        with self.lock:
            self.in_progress_players.discard(player_name)
            self.visited_players.add(player_name)
            self.players_completed_this_run += 1
            if MAX_PLAYERS_PER_RUN > 0 and self.players_completed_this_run >= MAX_PLAYERS_PER_RUN:
                self.stop.set()
            elif (
                MAX_PLAYERS_PER_RUN == 0
                and SMOKE_PLAYER_PAGE_LIMIT > 0
                and len(self.visited_players) >= SMOKE_PLAYER_PAGE_LIMIT
            ):
                self.stop.set()

    def complete_player_failure(self, player_name: str) -> None:
        with self.lock:
            self.in_progress_players.discard(player_name)

    def record_match(self, match: dict) -> str:
        """Assign match_id and store match under lock.

        Returns 'saved', 'duplicate', or 'updated'.
        """
        signature = match_signature(match)
        with self.lock:
            if signature in self.match_signatures:
                existing_index = self.signature_to_index.get(signature)
                if existing_index is not None and should_merge_final_ban_targets(
                    self.matches[existing_index],
                    match,
                ):
                    merged = merge_final_ban_targets(self.matches[existing_index], match)
                    validation_errors = validate_match_record(merged)
                    if validation_errors:
                        return "duplicate"
                    self.matches[existing_index] = merged
                    return "updated"
                return "duplicate"
            match["match_id"] = self.next_match_id
            self.next_match_id += 1
            self.match_signatures.add(signature)
            self.signature_to_index[signature] = len(self.matches)
            self.matches.append(match)
            return "saved"

    def save_all(self) -> None:
        with self.lock:
            self._save_raw_jsonl_unlocked()
            self._save_scraper_state_unlocked()
            match_count = len(self.matches)
            next_id = self.next_match_id

        print(f"Saved scraper state: match_count={match_count}, next_match_id={next_id}")

    def save_after_player(self) -> None:
        with self.lock:
            self._save_raw_jsonl_unlocked()
            try:
                self._save_scraper_state_unlocked()
            except PermissionError as exc:
                print(f"Warning: could not save scraper state (file locked?): {exc}")

    def _save_raw_jsonl_unlocked(self) -> None:
        path = Path(RAW_JSONL_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            for match in self.matches:
                file.write(json.dumps(match, ensure_ascii=False) + "\n")
        replace_with_retries(tmp_path, path)

    def _save_scraper_state_unlocked(self) -> None:
        SCRAPER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = SCRAPER_STATE_PATH.with_suffix(SCRAPER_STATE_PATH.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(
                {
                    "visited_players": sorted(self.visited_players),
                    "in_progress_players": sorted(self.in_progress_players),
                    "match_count": len(self.matches),
                    "next_match_id": self.next_match_id,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        replace_with_retries(tmp_path, SCRAPER_STATE_PATH)


def init_driver() -> webdriver.Chrome:
    global CHROME_DRIVER_PATH
    opts = Options()
    opts.page_load_strategy = "eager"
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1920,1080")
    with _chrome_driver_lock:
        if CHROME_DRIVER_PATH is None:
            CHROME_DRIVER_PATH = ChromeDriverManager().install()
        driver_path = CHROME_DRIVER_PATH
    svc = ChromeService(driver_path)
    new_driver = webdriver.Chrome(service=svc, options=opts)
    new_driver.set_page_load_timeout(45)
    new_driver.set_script_timeout(45)
    return new_driver


def is_browser_session_lost(exc: Exception) -> bool:
    if isinstance(exc, InvalidSessionIdException):
        return True
    msg = str(exc).lower()
    return (
        "invalid session id" in msg
        or "not connected to devtools" in msg
        or "browser has closed the connection" in msg
        or "disconnected" in msg
        or "timed out receiving message from renderer" in msg
        or "timeout: timed out receiving message" in msg
        or "renderer" in msg and "timeout" in msg
    )


def _get_el_select_display_text(select_el) -> str:
    for selector in (".el-select__selected-item", ".el-select__placeholder"):
        for part in select_el.find_elements(By.CSS_SELECTOR, selector):
            text = (part.text or "").strip()
            if text:
                return text
    return (select_el.text or "").strip()


def _click_visible_el_select_option(driver: webdriver.Chrome, option_text: str) -> None:
    clicked = driver.execute_script(
        """
        const target = arguments[0];
        const panels = Array.from(document.querySelectorAll('div.el-select-dropdown'));
        for (const panel of panels) {
            if (panel.style.display === 'none') continue;
            const items = panel.querySelectorAll('li.el-select-dropdown__item');
            for (const item of items) {
                const text = (item.textContent || '').trim();
                if (text === target) {
                    item.click();
                    return true;
                }
            }
        }
        return false;
        """,
        option_text,
    )
    if not clicked:
        raise TimeoutException(f"Option {option_text!r} not found in visible select dropdown")


def ensure_rank_server(driver: webdriver.Chrome, worker_id: int, server_name: str) -> bool:
    """Select the server filter on the rank page (Element Plus el-select)."""
    try:
        selects = WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".el-select"))
        )
        if len(selects) < 2:
            wlog(worker_id, f"Expected season + server selects, found {len(selects)}")
            server_select = selects[-1]
        else:
            server_select = selects[1]

        current = _get_el_select_display_text(server_select)
        if current == server_name:
            return True

        wlog(worker_id, f"Switching rank server from {current!r} to {server_name!r}")
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", server_select)
        time.sleep(0.2)
        driver.execute_script("arguments[0].click();", server_select)
        time.sleep(0.5)
        _click_visible_el_select_option(driver, server_name)

        def server_filter_applied(d: webdriver.Chrome) -> bool:
            current_selects = d.find_elements(By.CSS_SELECTOR, ".el-select")
            if len(current_selects) < 2:
                target = current_selects[-1] if current_selects else None
            else:
                target = current_selects[1]
            if target is None:
                return False
            return _get_el_select_display_text(target) == server_name

        WebDriverWait(driver, 15).until(server_filter_applied)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "hero-name"))
        )
        time.sleep(1)
        return True
    except Exception as exc:
        wlog(worker_id, f"Could not select rank server {server_name!r}: {exc}")
        return False


def navigate_to_rank_page(driver: webdriver.Chrome, worker_id: int) -> bool:
    driver.get(INITIAL_URL)
    WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    time.sleep(1)
    if not ensure_rank_server(driver, worker_id, RANK_SERVER):
        return False
    return True


def start_driver(worker_id: int = 0) -> webdriver.Chrome:
    new_driver = init_driver()
    if not navigate_to_rank_page(new_driver, worker_id):
        wlog(worker_id, f"Rank page loaded but server filter may not be {RANK_SERVER!r}")
    return new_driver


def ensure_match_detail_open(driver: webdriver.Chrome, worker_id: int, index: int) -> bool:
    """Open a battle row before parsing detail-only markers like FIRST PICK."""
    battle_rows = driver.find_elements(By.CSS_SELECTOR, "li.battle-info.win, li.battle-info.lose")
    if index >= len(battle_rows):
        wlog(worker_id, f"Skipping match {index}: row disappeared before opening detail")
        return False

    row = battle_rows[index]
    row_class = row.get_attribute("class") or ""
    if "open" in row_class.split():
        return True

    btn_details = row.find_elements(By.CLASS_NAME, "btn-detail")
    if not btn_details:
        wlog(worker_id, f"Skipping match {index}: detail button not found in row")
        return False

    try:
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn_details[0])
        time.sleep(0.2)
        driver.execute_script("arguments[0].click();", btn_details[0])
        WebDriverWait(driver, 10).until(
            lambda d: (
                len(rows := d.find_elements(By.CSS_SELECTOR, "li.battle-info.win, li.battle-info.lose"))
                > index
                and "open" in ((rows[index].get_attribute("class") or "").split())
            )
        )
    except TimeoutException:
        wlog(worker_id, f"Skipping match {index}: detail did not open in time")
        return False
    except StaleElementReferenceException:
        wlog(worker_id, f"Skipping match {index}: row changed while opening detail")
        return False

    return True


def extract_match_data(
    coordinator: ScraperCoordinator,
    soup: BeautifulSoup,
    index: int,
    *,
    source_player: str,
    worker_id: int,
) -> str:
    battle_info_elements = soup.select("li.battle-info.win, li.battle-info.lose")

    if len(battle_info_elements) < 1:
        return "invalid"

    if index >= len(battle_info_elements):
        wlog(worker_id, f"Skipping match {index}: row disappeared before parsing detail")
        return "invalid"

    battle = battle_info_elements[index]
    played_at, played_at_text = extract_played_at(battle)
    if MAX_MATCH_DAYS_ENABLED and MAX_MATCH_DAYS is not None and played_at is not None:
        if not is_match_within_days(played_at, MAX_MATCH_DAYS):
            label = played_at_text or played_at.isoformat(timespec="seconds")
            wlog(worker_id, f"Skipping match {index}: older than {MAX_MATCH_DAYS} day(s) ({label})")
            return "too_old"

    winner_side = "ally" if "win" in battle.get("class", []) else "enemy"

    my_team = battle.find("div", class_="my-team w-100")
    enemy_team = battle.find("div", class_="enemy-team w-100")

    first_pick_side = detect_first_pick_side_from_battle(battle)
    if first_pick_side is None:
        wlog(worker_id, "Skipping match: unable to detect visible FIRST PICK marker")
        return "invalid"

    ally_extraction = extract_team_draft(my_team)
    enemy_extraction = extract_team_draft(enemy_team)

    if ally_extraction.error:
        wlog(worker_id, f"Skipping match {index}: ally team parse error: {ally_extraction.error}")
        return "invalid"
    if enemy_extraction.error:
        wlog(worker_id, f"Skipping match {index}: enemy team parse error: {enemy_extraction.error}")
        return "invalid"

    ally_preban = ally_extraction.prebans
    enemy_preban = enemy_extraction.prebans
    ally_picks = ally_extraction.picks
    enemy_picks = enemy_extraction.picks

    if ally_extraction.final_banned_pick is None or enemy_extraction.final_banned_pick is None:
        wlog(
            worker_id,
            "Skipping match "
            f"{index}: missing final ban marker "
            f"(ally={ally_extraction.final_banned_pick}, enemy={enemy_extraction.final_banned_pick})",
        )
        return "invalid"

    draft = []
    ally_orders = FIRST_PICK_ORDERS.copy() if first_pick_side == "ally" else SECOND_PICK_ORDERS.copy()
    enemy_orders = FIRST_PICK_ORDERS.copy() if first_pick_side == "enemy" else SECOND_PICK_ORDERS.copy()

    for hero in reversed(ally_picks):
        draft.append(make_draft_entry(order=ally_orders.pop(0), side="ally", hero=hero))

    for hero in enemy_picks:
        draft.append(make_draft_entry(order=enemy_orders.pop(0), side="enemy", hero=hero))

    if len(draft) != 10:
        return "invalid"

    match = {
        "first_pick_side": first_pick_side,
        "winner_side": winner_side,
        "ally_preban": ally_preban,
        "enemy_preban": enemy_preban,
        ALLY_FINAL_BAN_TARGET: enemy_extraction.final_banned_pick,
        ENEMY_FINAL_BAN_TARGET: ally_extraction.final_banned_pick,
        "draft": draft,
    }
    match.update(extract_season_patch(battle))
    match.update(extract_warfare_rules(battle))
    match["source_player"] = source_player
    if played_at is not None:
        match["played_at"] = played_at.isoformat(timespec="seconds")
    if played_at_text:
        match["played_at_text"] = played_at_text

    validation_errors = validate_match_record({**match, "match_id": -1})
    if validation_errors:
        wlog(
            worker_id,
            f"Skipping match {index}: final ban validation failed: {validation_errors[0]}",
        )
        return "invalid"

    return coordinator.record_match(match)


def claim_next_player(driver: webdriver.Chrome, coordinator: ScraperCoordinator) -> str | None:
    """Return a claimed player name from the current rank page, or None if none available."""
    hero_elements = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, "hero-name"))
    )
    player_names = [element.text.strip() for element in hero_elements if element.text.strip()]
    for index, player_name in enumerate(player_names):
        if RANK_TOP_LIMIT > 0 and index >= RANK_TOP_LIMIT:
            break
        if coordinator.try_claim_player(player_name):
            return player_name
    return None


def click_player_on_rank_page(driver: webdriver.Chrome, player_name: str) -> bool:
    hero_elements = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, "hero-name"))
    )
    for element in hero_elements:
        if element.text.strip() == player_name:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            time.sleep(0.3)
            driver.execute_script("arguments[0].click();", element)
            time.sleep(5)
            return True
    return False


def process_player(
    driver: webdriver.Chrome,
    coordinator: ScraperCoordinator,
    *,
    player_name: str,
    worker_id: int,
) -> bool:
    """Scrape one player's matches. Returns True on normal completion, False on hard failure."""
    if not click_player_on_rank_page(driver, player_name):
        wlog(worker_id, f"Could not open player page for {player_name}")
        return False

    clicked_button_indices: list[int] = []
    matches_scraped = 0
    stop_player_for_age = False

    while matches_scraped < MAX_MATCHES_PER_PLAYER and not stop_player_for_age:
        try:
            btn_details = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "btn-detail"))
            )
            new_button_indices = [
                i for i in range(len(btn_details)) if i not in clicked_button_indices
            ]

            if not new_button_indices:
                break

            for index in new_button_indices:
                try:
                    if not ensure_match_detail_open(driver, worker_id, index):
                        clicked_button_indices.append(index)
                        continue
                    page_source = driver.page_source
                    soup = BeautifulSoup(page_source, "html.parser")
                    result = extract_match_data(
                        coordinator,
                        soup,
                        index,
                        source_player=player_name,
                        worker_id=worker_id,
                    )
                    matches_scraped += 1
                    clicked_button_indices.append(index)

                    if result == "too_old":
                        wlog(
                            worker_id,
                            f"Reached matches older than {MAX_MATCH_DAYS} day(s) for {player_name}",
                        )
                        stop_player_for_age = True
                        break

                    if matches_scraped >= MAX_MATCHES_PER_PLAYER:
                        break
                except Exception as exc:
                    wlog(worker_id, f"Error clicking button: {exc}")
                    if is_browser_session_lost(exc):
                        raise

            if stop_player_for_age:
                break

            load_more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CLASS_NAME, "loadMoreBtn"))
            )
            load_more_button.click()
            time.sleep(2)

        except Exception as exc:
            wlog(worker_id, f"Error in player loop: {exc}")
            if is_browser_session_lost(exc):
                return False
            break

    if not navigate_to_rank_page(driver, worker_id):
        wlog(worker_id, f"Could not return to {RANK_SERVER!r} rank page after {player_name}")
        return False
    return True


def load_more_rank_players(driver: webdriver.Chrome, worker_id: int) -> bool:
    try:
        load_more_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "loadMoreBtn"))
        )
        driver.execute_script("arguments[0].click();", load_more_button)
        time.sleep(2)
        return True
    except Exception as exc:
        wlog(worker_id, f"No more players to load on rank page: {exc}")
        return False


def run_worker(worker_id: int, coordinator: ScraperCoordinator) -> None:
    driver: webdriver.Chrome | None = None
    players_processed = 0
    rank_load_failures = 0

    try:
        driver = start_driver(worker_id)
        wlog(worker_id, f"started (rank server={RANK_SERVER!r})")

        while not coordinator.should_stop():
            if players_processed >= RESTART_THRESHOLD:
                wlog(worker_id, f"Restarting browser after {RESTART_THRESHOLD} players")
                try:
                    driver.quit()
                except Exception:
                    pass
                driver = start_driver(worker_id)
                players_processed = 0

            player_name = claim_next_player(driver, coordinator)
            if player_name is None:
                if RANK_TOP_LIMIT > 0:
                    wlog(worker_id, f"Rank top {RANK_TOP_LIMIT} exhausted; stopping worker")
                    break
                if not load_more_rank_players(driver, worker_id):
                    rank_load_failures += 1
                    if rank_load_failures >= 5:
                        wlog(worker_id, "Rank page exhausted; stopping worker")
                        break
                    continue
                rank_load_failures = 0
                continue

            rank_load_failures = 0
            wlog(worker_id, f"processing: {player_name}")

            try:
                completed = process_player(
                    driver,
                    coordinator,
                    player_name=player_name,
                    worker_id=worker_id,
                )
            except Exception as exc:
                wlog(worker_id, f"Hard failure while processing {player_name}: {exc}")
                coordinator.complete_player_failure(player_name)
                if is_browser_session_lost(exc):
                    try:
                        driver.quit()
                    except Exception:
                        pass
                    driver = start_driver(worker_id)
                continue

            if completed:
                coordinator.complete_player_success(player_name)
                coordinator.save_after_player()
                players_processed += 1
                wlog(worker_id, f"finished: {player_name}")
            else:
                coordinator.complete_player_failure(player_name)
                wlog(worker_id, f"failed (will retry later): {player_name}")
                try:
                    driver.quit()
                except Exception:
                    pass
                driver = start_driver(worker_id)

    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass
        wlog(worker_id, "stopped")


def main() -> None:
    coordinator = ScraperCoordinator()
    coordinator.load_existing_matches()
    coordinator.load_scraper_state()

    if MAX_MATCH_DAYS_ENABLED and MAX_MATCH_DAYS is not None:
        print(f"Date filter enabled: only matches within the last {MAX_MATCH_DAYS} day(s)")
    else:
        print("Date filter disabled (set E7_GET_MATCHES_MAX_DAYS_ENABLED=1 to enable)")
    print(f"Starting {WORKER_COUNT} worker(s) on rank server {RANK_SERVER!r}")
    if RANK_TOP_LIMIT > 0:
        print(f"Rank top limit enabled: first {RANK_TOP_LIMIT} player(s) on leaderboard")
    if MAX_PLAYERS_PER_RUN > 0:
        print(f"Run player limit enabled: stop after {MAX_PLAYERS_PER_RUN} completed player(s)")
    if REVISIT_VISITED_PLAYERS:
        print("Revisit enabled: already-visited players may be scraped again")

    threads: list[threading.Thread] = []
    try:
        for worker_id in range(WORKER_COUNT):
            thread = threading.Thread(
                target=run_worker,
                args=(worker_id + 1, coordinator),
                name=f"scraper-worker-{worker_id + 1}",
                daemon=False,
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("Manually interrupted; stopping workers...")
        coordinator.stop.set()
        for thread in threads:
            thread.join(timeout=30)
    finally:
        coordinator.save_all()


if __name__ == "__main__":
    main()
