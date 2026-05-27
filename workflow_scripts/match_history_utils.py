"""Shared helpers for Epic Seven RTA match history (JSONL raw format)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dateutil.relativedelta import relativedelta

SIDES = frozenset({"ally", "enemy"})
RAW_JSONL_PATH = Path("data/epic7_match_history_raw.jsonl")
POSITION_BUCKETS = frozenset(
    {"1", "2_3", "4", "5_protected", "6_protected", "7", "8_9", "10"}
)


def get_position_bucket(order: int) -> str:
    if order == 1:
        return "1"
    if order in (2, 3):
        return "2_3"
    if order == 4:
        return "4"
    if order == 5:
        return "5_protected"
    if order == 6:
        return "6_protected"
    if order == 7:
        return "7"
    if order in (8, 9):
        return "8_9"
    if order == 10:
        return "10"
    raise ValueError(f"Invalid draft order: {order}")


def make_draft_entry(*, order: int, side: str, hero: str) -> dict[str, Any]:
    return {
        "order": order,
        "side": side,
        "hero": hero,
        "position_bucket": get_position_bucket(order),
    }


def enrich_match_draft(match: dict[str, Any]) -> dict[str, Any]:
    enriched_draft: list[dict[str, Any]] = []
    for entry in match["draft"]:
        order = int(entry["order"])
        enriched = dict(entry)
        enriched["position_bucket"] = get_position_bucket(order)
        enriched_draft.append(enriched)
    return {**match, "draft": enriched_draft}


def hero_code_from_li(hero_li) -> str | None:
    hero_img = hero_li.find("img")
    if not hero_img:
        return None
    code = hero_img.get("alt")
    if not code or code == "Unknown":
        return None
    return code


def extract_preban_and_picks(team_div) -> tuple[list[str], list[str]]:
    """Stove RTA: prebans are li.preban-hero; draft picks are li.pick-hero (may also have ban)."""
    if not team_div:
        return [], []
    preban = []
    for li in team_div.select("li.preban-hero"):
        code = hero_code_from_li(li)
        if code:
            preban.append(code)
    picks = []
    for li in team_div.select("li.pick-hero"):
        code = hero_code_from_li(li)
        if code:
            picks.append(code)
    return preban, picks


def detect_first_pick_side(ally_team_div, enemy_team_div) -> str | None:
    """Detect the visible FIRST PICK marker from Stove team panes.

    Stove renders `em.firstpick` on both panes in some layouts, but only the
    active first-pick side carries the `show` class.
    """
    ally_visible = bool(ally_team_div and ally_team_div.select_one("em.firstpick.show"))
    enemy_visible = bool(enemy_team_div and enemy_team_div.select_one("em.firstpick.show"))
    if ally_visible != enemy_visible:
        return "ally" if ally_visible else "enemy"

    # Fallback for older markup where only one side contains the marker.
    ally_marker = bool(ally_team_div and ally_team_div.select_one("em.firstpick"))
    enemy_marker = bool(enemy_team_div and enemy_team_div.select_one("em.firstpick"))
    if ally_marker != enemy_marker:
        return "ally" if ally_marker else "enemy"

    return None


def detect_first_pick_side_from_battle(battle) -> str | None:
    """Detect first-pick side from both summary and expanded detail markup."""
    for ally_selector, enemy_selector in (
        ("div.my-team.w-100", "div.enemy-team.w-100"),
        ("div.my-team-detail", "div.enemy-team-detail"),
    ):
        first_pick_side = detect_first_pick_side(
            battle.select_one(ally_selector),
            battle.select_one(enemy_selector),
        )
        if first_pick_side is not None:
            return first_pick_side
    return None


def extract_season_patch(battle) -> dict[str, str]:
    """Return season/patch when present on the battle card."""
    extra: dict[str, str] = {}
    season_el = battle.select_one("p.season-name")
    if season_el and season_el.get_text(strip=True):
        extra["season"] = season_el.get_text(strip=True)
    patch_el = battle.select_one("span.patch-version")
    if patch_el and patch_el.get_text(strip=True):
        extra["patch"] = patch_el.get_text(strip=True)
    return extra


def extract_warfare_rules(battle) -> dict[str, str]:
    """Return warfare_rules from Stove battle summary when present."""
    rule_el = battle.select_one("p.opening-rule-block")
    if rule_el is None:
        return {}

    raw_text = rule_el.get_text(" ", strip=True)
    if not raw_text:
        return {}

    if raw_text.lower().startswith("warfare rules"):
        _, _, value = raw_text.partition(":")
        rule = value.strip()
    else:
        rule = raw_text

    if not rule:
        return {}
    return {"warfare_rules": rule}


_RELATIVE_TIME_RE = re.compile(
    r"^(?:(?P<just>just now)|(?:(?P<a_an>a|an)\s+)?(?P<num>\d+)?\s*"
    r"(?P<unit>second|minute|hour|day|week|month|year)s?\s+ago)$",
    re.IGNORECASE,
)


def parse_relative_time_text(text: str, *, now: datetime | None = None) -> datetime | None:
    """Parse Stove relative timestamps such as 'a day ago' or '3 hours ago'."""
    normalized = text.strip()
    if not normalized:
        return None

    match = _RELATIVE_TIME_RE.match(normalized)
    if not match:
        return None

    reference = now or datetime.now()
    if match.group("just"):
        return reference

    amount = int(match.group("num") or 1)
    unit = match.group("unit").lower()
    if unit in {"second", "minute", "hour", "day", "week"}:
        delta = timedelta(**{f"{unit}s": amount})
        return reference - delta
    if unit == "month":
        return reference - relativedelta(months=amount)
    if unit == "year":
        return reference - relativedelta(years=amount)
    return None


def extract_played_at(battle, *, now: datetime | None = None) -> tuple[datetime | None, str | None]:
    """Return parsed played-at time and the raw relative text from a battle card."""
    info = battle.select_one("div.battle-result-info")
    if not info:
        return None, None

    for paragraph in info.find_all("p", recursive=False):
        if paragraph.get("class"):
            continue
        raw_text = paragraph.get_text(" ", strip=True)
        played_at = parse_relative_time_text(raw_text, now=now)
        if played_at is not None:
            return played_at, raw_text
    return None, None


def is_match_within_days(
    played_at: datetime,
    max_days: int,
    *,
    now: datetime | None = None,
) -> bool:
    cutoff = (now or datetime.now()) - timedelta(days=max_days)
    return played_at >= cutoff


def validate_match_record(match: dict[str, Any]) -> list[str]:
    """Return a list of validation error messages (empty if valid)."""
    errors: list[str] = []
    required = (
        "match_id",
        "first_pick_side",
        "winner_side",
        "ally_preban",
        "enemy_preban",
        "draft",
    )
    for key in required:
        if key not in match:
            errors.append(f"match {match.get('match_id', '?')}: missing field '{key}'")
            return errors

    fps = match["first_pick_side"]
    ws = match["winner_side"]
    if fps not in SIDES:
        errors.append(f"match {match['match_id']}: invalid first_pick_side '{fps}'")
    if ws not in SIDES:
        errors.append(f"match {match['match_id']}: invalid winner_side '{ws}'")

    for field in ("ally_preban", "enemy_preban"):
        bans = match[field]
        if not isinstance(bans, list):
            errors.append(f"match {match['match_id']}: {field} must be a list")
            continue
        if len(bans) != len(set(bans)):
            errors.append(f"match {match['match_id']}: duplicate heroes in {field}")

    draft = match.get("draft")
    if not isinstance(draft, list):
        errors.append(f"match {match['match_id']}: draft must be a list")
        return errors

    orders: list[int] = []
    ally_picks = 0
    enemy_picks = 0
    drafted_heroes: list[str] = []

    for entry in draft:
        if not isinstance(entry, dict):
            errors.append(f"match {match['match_id']}: draft entry must be an object")
            continue
        side = entry.get("side")
        hero = entry.get("hero")
        order = entry.get("order")
        position_bucket = entry.get("position_bucket")
        if side not in SIDES:
            errors.append(f"match {match['match_id']}: invalid draft side '{side}'")
        if not hero:
            errors.append(f"match {match['match_id']}: draft entry missing hero")
        else:
            drafted_heroes.append(hero)
        if order is not None:
            order_int = int(order)
            orders.append(order_int)
            expected_bucket = get_position_bucket(order_int)
            if position_bucket is None:
                errors.append(
                    f"match {match['match_id']}: draft order {order_int} missing position_bucket"
                )
            elif position_bucket != expected_bucket:
                errors.append(
                    f"match {match['match_id']}: draft order {order_int} has "
                    f"position_bucket '{position_bucket}', expected '{expected_bucket}'"
                )
            elif position_bucket not in POSITION_BUCKETS:
                errors.append(
                    f"match {match['match_id']}: invalid position_bucket '{position_bucket}'"
                )
        if side == "ally":
            ally_picks += 1
        elif side == "enemy":
            enemy_picks += 1

    if len(orders) != len(set(orders)):
        errors.append(f"match {match['match_id']}: duplicate draft order values")
    if len(orders) != 10 or sorted(orders) != list(range(1, 11)):
        errors.append(f"match {match['match_id']}: draft orders must be 1..10")
    if ally_picks != 5 or enemy_picks != 5:
        errors.append(
            f"match {match['match_id']}: expected 5 ally and 5 enemy picks, "
            f"got {ally_picks}/{enemy_picks}"
        )
    if len(drafted_heroes) != len(set(drafted_heroes)):
        errors.append(f"match {match['match_id']}: duplicate heroes in draft")

    order_one = next((e for e in draft if e.get("order") == 1), None)
    if order_one and fps in SIDES and order_one.get("side") != fps:
        errors.append(
            f"match {match['match_id']}: first_pick_side '{fps}' "
            f"does not match draft order 1 side '{order_one.get('side')}'"
        )

    ban_set = set(match.get("ally_preban") or []) | set(match.get("enemy_preban") or [])
    overlap = ban_set & set(drafted_heroes)
    if overlap:
        errors.append(
            f"match {match['match_id']}: prebanned heroes appear in draft: {sorted(overlap)}"
        )

    return errors
