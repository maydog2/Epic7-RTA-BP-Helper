export type FirstPickTeam = "My Team" | "Enemy Team";

export type WarfareRule = "ANY" | "Support" | "Offense" | "Defense" | "Resistance";

export type Hero = {
  code: string;
  name: string;
  name_en?: string;
  name_zh?: string;
  role: string;
  element: string;
  appearance_count: number;
  avatar_url: string;
  element_icon_url: string;
  role_icon_url: string;
};

export type RecommendationResponse = {
  top_10_heroes: string[];
  /** Pick phase: softmax×100 = model % for next lock. Ban phase: renorm among enemy picks. Preban: % share of all historical pre-ban occurrences (heroes sum to ~100% across full data). */
  top_10_rates?: number[];
  recommendations?: PrebanRecommendation[];
  /** After both teams lock 5: ranked enemy bans; rates renorm softmax among enemy picks only (~100%). */
  phase?: "preban" | "pick" | "ban";
};

export type PrebanRecommendation = {
  hero_id: string;
  normalized_preban_rate: number;
};
