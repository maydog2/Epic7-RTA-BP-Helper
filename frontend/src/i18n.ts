import type { Hero } from "./types";

export type AppLanguage = "en" | "zh";

export const LANGUAGE_STORAGE_KEY = "e7-bp-helper-language";

const messages = {
  en: {
    settings: "Settings",
    language: "Language",
    languageEn: "English",
    languageZh: "中文",
    allyFirst: "Ally first",
    enemyFirst: "Enemy first",
    rememberPreban: "Remember Preban",
    warfareRules: "Warfare rules",
    warfareRuleANY: "None",
    warfareRuleSupport: "Support",
    warfareRuleOffense: "Offense",
    warfareRuleDefense: "Defense",
    warfareRuleResistance: "Resistance",
    draft: "Draft",
    undo: "Undo",
    preban: "Preban",
    pick: "Pick",
    ally: "Ally",
    enemy: "Enemy",
    emptySlot: "Empty slot",
    allyPrebanSlot: "Ally preban slot",
    enemyPrebanSlot: "Enemy preban slot",
    banned: "Banned",
    heroPicker: "Hero Picker",
    shown: "shown",
    tagged: "tagged",
    searchHero: "Search hero",
    all: "All",
    elementFilters: "Element filters",
    roleFilters: "Role filters",
    banPhaseCompleted: "Ban phase completed",
    banSuggestions: "Ban suggestions",
    prebanSuggestions: "Preban suggestions",
    pickSuggestions: "Pick suggestions",
    loadingSuggestions: "Loading suggestions…",
    completed: "completed",
    noPrebanData: "No preban data",
    noBanData: "No ban data",
    draftComplete: "Draft complete",
    loadHeroesFailed: "Unable to load heroes.",
    loadPickSuggestionsFailed: "Failed to load pick suggestions",
    elementFire: "Fire",
    elementIce: "Ice",
    elementEarth: "Earth",
    elementLight: "Light",
    elementDark: "Dark",
    roleWarrior: "Warrior",
    roleKnight: "Knight",
    roleMage: "Mage",
    roleRanger: "Ranger",
    roleAssassin: "Assassin",
    roleManauser: "Manauser",
  },
  zh: {
    settings: "设置",
    language: "语言",
    languageEn: "English",
    languageZh: "中文",
    allyFirst: "我方先手",
    enemyFirst: "对方先手",
    rememberPreban: "记住预禁",
    warfareRules: "开战规则",
    warfareRuleANY: "无",
    warfareRuleSupport: "支援",
    warfareRuleOffense: "攻击",
    warfareRuleDefense: "防御",
    warfareRuleResistance: "抵抗",
    draft: "选人",
    undo: "撤销",
    preban: "预禁",
    pick: "Pick",
    ally: "我方",
    enemy: "对方",
    emptySlot: "空位",
    allyPrebanSlot: "我方预禁位",
    enemyPrebanSlot: "对方预禁位",
    banned: "已禁用",
    heroPicker: "英雄选择",
    shown: "显示",
    tagged: "已标注",
    searchHero: "搜索英雄",
    all: "全部",
    elementFilters: "属性筛选",
    roleFilters: "职业筛选",
    banPhaseCompleted: "禁用阶段已完成",
    banSuggestions: "禁用推荐",
    prebanSuggestions: "预禁推荐",
    pickSuggestions: "选人推荐",
    loadingSuggestions: "加载推荐中…",
    completed: "已完成",
    noPrebanData: "无预禁数据",
    noBanData: "无禁用数据",
    draftComplete: "选人已完成",
    loadHeroesFailed: "无法加载英雄列表。",
    loadPickSuggestionsFailed: "加载选人推荐失败",
    elementFire: "火",
    elementIce: "冰",
    elementEarth: "木",
    elementLight: "光",
    elementDark: "暗",
    roleWarrior: "战士",
    roleKnight: "骑士",
    roleMage: "法师",
    roleRanger: "射手",
    roleAssassin: "刺客",
    roleManauser: "精灵使",
  },
} as const;

export type MessageKey = keyof typeof messages.en;

export function getStoredLanguage(): AppLanguage {
  if (typeof window === "undefined") {
    return "en";
  }
  return window.localStorage.getItem(LANGUAGE_STORAGE_KEY) === "zh" ? "zh" : "en";
}

export function storeLanguage(language: AppLanguage): void {
  window.localStorage.setItem(LANGUAGE_STORAGE_KEY, language);
}

export function t(language: AppLanguage, key: MessageKey): string {
  return messages[language][key];
}

export function getHeroDisplayName(hero: Hero, language: AppLanguage): string {
  if (language === "zh") {
    return hero.name_zh || hero.name_en || hero.name || hero.code;
  }
  return hero.name_en || hero.name || hero.name_zh || hero.code;
}

export function localizeHeroes(heroes: Hero[], language: AppLanguage): Hero[] {
  return heroes.map((hero) => ({
    ...hero,
    name: getHeroDisplayName(hero, language),
  }));
}

export function localizeElement(element: string, language: AppLanguage): string {
  const keyMap: Record<string, MessageKey> = {
    fire: "elementFire",
    ice: "elementIce",
    earth: "elementEarth",
    light: "elementLight",
    dark: "elementDark",
  };
  const key = keyMap[element];
  return key ? t(language, key) : element;
}

export function localizeRole(role: string, language: AppLanguage): string {
  const keyMap: Record<string, MessageKey> = {
    warrior: "roleWarrior",
    knight: "roleKnight",
    mage: "roleMage",
    ranger: "roleRanger",
    assassin: "roleAssassin",
    manauser: "roleManauser",
  };
  const key = keyMap[role];
  return key ? t(language, key) : role;
}

export function heroMatchesSearch(hero: Hero, query: string, language: AppLanguage): boolean {
  const normalized = query.trim().toLowerCase();
  if (!normalized) {
    return true;
  }
  const candidates = [
    getHeroDisplayName(hero, language),
    hero.name_en ?? "",
    hero.name_zh ?? "",
    hero.name,
    hero.code,
  ];
  return candidates.some((value) => value.toLowerCase().includes(normalized));
}
