import { getHeroDisplayName, type AppLanguage, type MessageKey } from "./i18n";
import type { FirstPickTeam, Hero, WarfareRule } from "./types";

export type Team = "user" | "enemy";

export type CurrentDraftStep = {
  phase: "preban" | "pick";
  team: Team;
  slotIndex: number;
} | null;

export type DraftPick = {
  team: Team;
  code: string;
};

export type PrebanMemoryMode = "shared" | "split";

export type AllyPrebanPresets = Record<FirstPickTeam, DraftPick[]>;

export const MAX_TEAM_SIZE = 5;
export const MAX_PREBAN_SIZE = 4;
export const PREBAN_SUGGESTION_DISPLAY_SIZE = 10;
/** Fetch one extra so a picked suggestion can be replaced without refetching. */
export const PREBAN_SUGGESTION_POOL_SIZE = PREBAN_SUGGESTION_DISPLAY_SIZE + 1;
export const WARFARE_RULE_OPTIONS: WarfareRule[] = ["ANY", "Support", "Offense", "Defense", "Resistance"];
/** Pick slot index 2 = third lock; cannot be chosen as ban target */
export const BAN_PROTECTED_SLOT_INDEX = 2;
export const PICK_ORDER_PATTERN = [
  "first",
  "second",
  "second",
  "first",
  "first",
  "second",
  "second",
  "first",
  "first",
  "second",
] as const;

export const EMPTY_ALLY_PREBAN_PRESETS: AllyPrebanPresets = {
  "My Team": [],
  "Enemy Team": [],
};

/** RTA: ally always prebans first (2 slots), then enemy (2 slots), regardless of first pick. */
export const PREBAN_ORDER: Team[] = ["user", "user", "enemy", "enemy"];

export const ELEMENT_FILTER_ORDER = ["fire", "ice", "earth", "light", "dark"] as const;
export const ROLE_FILTER_ORDER = ["warrior", "knight", "mage", "ranger", "assassin", "manauser"] as const;

const BAN_PRIORITY_LABEL_KEYS: MessageKey[] = [
  "banPriorityTop",
  "banPriorityHigh",
  "banPriorityMedium",
  "banPriorityLow",
];

export function picksByTeam(picks: DraftPick[], team: Team): DraftPick[] {
  return picks.filter((pick) => pick.team === team);
}

export function mergePrebanPicks(allyPrebans: DraftPick[], enemyPrebans: DraftPick[]): DraftPick[] {
  return [...allyPrebans, ...enemyPrebans];
}

export function prebanPicksFromUserPresets(userPresets: DraftPick[]): DraftPick[] {
  return userPresets.map((pick) => ({ team: "user" as const, code: pick.code }));
}

export function sortByPredefinedOrder<T extends string>(values: Iterable<T>, order: readonly T[]): T[] {
  const orderIndex = new Map(order.map((value, index) => [value, index]));
  return [...new Set(values)].sort(
    (left, right) =>
      (orderIndex.get(left) ?? Number.MAX_SAFE_INTEGER) - (orderIndex.get(right) ?? Number.MAX_SAFE_INTEGER),
  );
}

export function banPriorityLabelKey(index: number): MessageKey {
  return BAN_PRIORITY_LABEL_KEYS[Math.min(index, BAN_PRIORITY_LABEL_KEYS.length - 1)];
}

/** Same hero may appear on ally preban and enemy preban; only block duplicate on the side that is picking next. */
function isPrebanDuplicateForTeam(team: Team, code: string, prebanPicks: DraftPick[]): boolean {
  return prebanPicks.some((p) => p.team === team && p.code === code);
}

export function isHeroUnavailableForNextPick(
  code: string,
  prebanPicks: DraftPick[],
  globallyUsed: Set<string>,
  prebanOrder: Team[],
): boolean {
  if (prebanPicks.length >= MAX_PREBAN_SIZE) {
    return globallyUsed.has(code);
  }
  const nextTeam = prebanOrder[prebanPicks.length];
  return isPrebanDuplicateForTeam(nextTeam, code, prebanPicks);
}

export function getHeroName(heroLookup: Map<string, Hero>, code: string, language: AppLanguage): string {
  const hero = heroLookup.get(code);
  return hero ? getHeroDisplayName(hero, language) : code;
}

export function getPickTeam(firstPickTeam: FirstPickTeam, pickIndex: number): Team {
  const firstTeam: Team = firstPickTeam === "My Team" ? "user" : "enemy";
  const secondTeam: Team = firstTeam === "user" ? "enemy" : "user";
  return PICK_ORDER_PATTERN[pickIndex] === "first" ? firstTeam : secondTeam;
}

export function isBanProtectedHero(code: string, userPicks: DraftPick[], enemyPicks: DraftPick[]): boolean {
  const ally = userPicks[BAN_PROTECTED_SLOT_INDEX];
  const foe = enemyPicks[BAN_PROTECTED_SLOT_INDEX];
  return Boolean((ally && ally.code === code) || (foe && foe.code === code));
}
