import type { FirstPickTeam, Hero, RecommendationResponse, WarfareRule } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";
const RECOMMENDATION_CACHE_MAX = 128;

const recommendationCache = new Map<string, RecommendationResponse>();
const recommendationInFlight = new Map<string, Promise<RecommendationResponse>>();

async function requestJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`);
  const body = await response.json();

  if (!response.ok) {
    throw new Error(body.message ?? "Request failed");
  }

  return body as T;
}

function normalizeWarfareRules(rule?: WarfareRule): WarfareRule {
  return rule ?? "ANY";
}

export function buildRecommendationCacheKey(params: {
  userPicks: string[];
  enemyPicks: string[];
  allyPreban?: string[];
  enemyPreban?: string[];
  firstPickTeam: FirstPickTeam;
  warfareRules?: WarfareRule;
}): string {
  const allyPreban = [...(params.allyPreban ?? [])].sort();
  const enemyPreban = [...(params.enemyPreban ?? [])].sort();
  return JSON.stringify([
    params.userPicks,
    params.enemyPicks,
    allyPreban,
    enemyPreban,
    params.firstPickTeam,
    normalizeWarfareRules(params.warfareRules),
  ]);
}

function storeRecommendationCache(key: string, payload: RecommendationResponse): void {
  if (recommendationCache.has(key)) {
    recommendationCache.delete(key);
  }
  recommendationCache.set(key, payload);
  while (recommendationCache.size > RECOMMENDATION_CACHE_MAX) {
    const oldestKey = recommendationCache.keys().next().value;
    if (oldestKey === undefined) {
      break;
    }
    recommendationCache.delete(oldestKey);
  }
}

export function clearRecommendationCache(): void {
  recommendationCache.clear();
  recommendationInFlight.clear();
}

export async function fetchHeroes(): Promise<Hero[]> {
  const data = await requestJson<{ heroes: Hero[] }>("/api/heroes");
  return data.heroes.map((hero) => ({
    ...hero,
    avatar_url:
      API_BASE_URL && hero.avatar_url.startsWith("/")
        ? `${API_BASE_URL}${hero.avatar_url}`
        : hero.avatar_url,
    element_icon_url:
      API_BASE_URL && hero.element_icon_url.startsWith("/")
        ? `${API_BASE_URL}${hero.element_icon_url}`
        : hero.element_icon_url,
    role_icon_url:
      API_BASE_URL && hero.role_icon_url.startsWith("/")
        ? `${API_BASE_URL}${hero.role_icon_url}`
        : hero.role_icon_url,
  }));
}

export async function fetchRecommendation(params: {
  userPicks: string[];
  enemyPicks: string[];
  allyPreban?: string[];
  enemyPreban?: string[];
  firstPickTeam: FirstPickTeam;
  warfareRules?: WarfareRule;
}): Promise<RecommendationResponse> {
  const cacheKey = buildRecommendationCacheKey(params);
  const cached = recommendationCache.get(cacheKey);
  if (cached) {
    return cached;
  }

  const inFlight = recommendationInFlight.get(cacheKey);
  if (inFlight) {
    return inFlight;
  }

  const searchParams = new URLSearchParams({
    user_picks: params.userPicks.join(","),
    enemy_picks: params.enemyPicks.join(","),
    ally_preban: (params.allyPreban ?? []).join(","),
    enemy_preban: (params.enemyPreban ?? []).join(","),
    first_pick_team: params.firstPickTeam,
    warfare_rules: normalizeWarfareRules(params.warfareRules),
  });

  const request = requestJson<RecommendationResponse>(`/api/recommend?${searchParams}`)
    .then((data) => {
      storeRecommendationCache(cacheKey, data);
      return data;
    })
    .finally(() => {
      recommendationInFlight.delete(cacheKey);
    });

  recommendationInFlight.set(cacheKey, request);
  return request;
}

export type PrebanSide = "user" | "enemy";

export async function fetchPrebanRecommendation(params: {
  excludedHeroes: string[];
  topK?: number;
  prebanSide?: PrebanSide;
  /** Ally first / Enemy first — filters historical stats by first_pick_side. */
  firstPickTeam: FirstPickTeam;
}): Promise<RecommendationResponse> {
  const searchParams = new URLSearchParams({
    excluded_heroes: params.excludedHeroes.join(","),
    top_k: String(params.topK ?? 10),
    first_pick_team: params.firstPickTeam,
  });

  if (params.prebanSide) {
    searchParams.set("preban_side", params.prebanSide);
  }

  return requestJson<RecommendationResponse>(`/api/preban_recommend?${searchParams}`);
}
