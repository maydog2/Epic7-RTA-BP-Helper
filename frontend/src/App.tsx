import { useEffect, useMemo, useState } from "react";
import { fetchHeroes, fetchPrebanRecommendation, fetchRecommendation, type PrebanSide } from "./api";
import {
  getHeroDisplayName,
  getStoredLanguage,
  heroMatchesSearch,
  localizeElement,
  localizeHeroes,
  localizeRole,
  storeLanguage,
  t,
  type AppLanguage,
  type MessageKey,
} from "./i18n";
import type { FirstPickTeam, Hero, RecommendationResponse, WarfareRule } from "./types";

type Team = "user" | "enemy";

type CurrentDraftStep = {
  phase: "preban" | "pick";
  team: Team;
  slotIndex: number;
} | null;

const MAX_TEAM_SIZE = 5;
const MAX_PREBAN_SIZE = 4;
const PREBAN_SUGGESTION_POOL_SIZE = 11;
const WARFARE_RULE_OPTIONS: WarfareRule[] = ["ANY", "Support", "Offense", "Defense", "Resistance"];
/** Pick slot index 2 = third lock; cannot be chosen as ban target */
const BAN_PROTECTED_SLOT_INDEX = 2;
const PICK_ORDER_PATTERN = ["first", "second", "second", "first", "first", "second", "second", "first", "first", "second"] as const;

type DraftPick = {
  team: Team;
  code: string;
};

type PrebanMemoryMode = "shared" | "split";

type AllyPrebanPresets = Record<FirstPickTeam, DraftPick[]>;

const EMPTY_ALLY_PREBAN_PRESETS: AllyPrebanPresets = {
  "My Team": [],
  "Enemy Team": [],
};

function extractUserPrebans(prebanPicks: DraftPick[]): DraftPick[] {
  return prebanPicks.filter((pick) => pick.team === "user");
}

function prebanPicksFromUserPresets(userPresets: DraftPick[]): DraftPick[] {
  return userPresets.map((pick) => ({ team: "user" as const, code: pick.code }));
}

/** RTA: ally always prebans first (2 slots), then enemy (2 slots), regardless of first pick. */
const PREBAN_ORDER: Team[] = ["user", "user", "enemy", "enemy"];

const ELEMENT_FILTER_ORDER = ["fire", "ice", "earth", "light", "dark"] as const;
const ROLE_FILTER_ORDER = ["warrior", "knight", "mage", "ranger", "assassin", "manauser"] as const;

function sortByPredefinedOrder<T extends string>(values: Iterable<T>, order: readonly T[]): T[] {
  const orderIndex = new Map(order.map((value, index) => [value, index]));
  return [...new Set(values)].sort(
    (left, right) =>
      (orderIndex.get(left) ?? Number.MAX_SAFE_INTEGER) - (orderIndex.get(right) ?? Number.MAX_SAFE_INTEGER),
  );
}

/** Same hero may appear on ally preban and enemy preban; only block duplicate on the side that is picking next. */
function isPrebanDuplicateForTeam(team: Team, code: string, prebanPicks: DraftPick[]): boolean {
  return prebanPicks.some((p) => p.team === team && p.code === code);
}

function isHeroUnavailableForNextPick(
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

function getHeroName(heroLookup: Map<string, Hero>, code: string, language: AppLanguage): string {
  const hero = heroLookup.get(code);
  return hero ? getHeroDisplayName(hero, language) : code;
}

function getPickTeam(firstPickTeam: FirstPickTeam, pickIndex: number): Team {
  const firstTeam: Team = firstPickTeam === "My Team" ? "user" : "enemy";
  const secondTeam: Team = firstTeam === "user" ? "enemy" : "user";
  return PICK_ORDER_PATTERN[pickIndex] === "first" ? firstTeam : secondTeam;
}

function isBanProtectedHero(code: string, userPicks: DraftPick[], enemyPicks: DraftPick[]): boolean {
  const ally = userPicks[BAN_PROTECTED_SLOT_INDEX];
  const foe = enemyPicks[BAN_PROTECTED_SLOT_INDEX];
  return Boolean((ally && ally.code === code) || (foe && foe.code === code));
}

function PickBanOverlay(props: { label: string }) {
  return (
    <span className="pick-slot-ban-badge" role="img" aria-label={props.label}>
      <svg viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="12" r="9.25" stroke="currentColor" strokeWidth="2" />
        <path d="M7 17 L17 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      </svg>
    </span>
  );
}

function HeroAvatar(props: { hero: Hero; displayName: string; size?: "small" | "large" }) {
  const [imageFailed, setImageFailed] = useState(!props.hero.avatar_url);
  const initials = props.displayName
    .split(/\s+/)
    .slice(0, 2)
    .map((part) => part[0])
    .join("")
    .toUpperCase();

  if (!props.hero.avatar_url || imageFailed) {
    return (
      <span className={`avatar-fallback ${props.size ?? "large"}`} title={props.displayName}>
        {initials}
      </span>
    );
  }

  return (
    <img
      className={`hero-avatar ${props.size ?? "large"}`}
      src={props.hero.avatar_url}
      alt={props.displayName}
      title={props.displayName}
      loading="lazy"
      onError={() => setImageFailed(true)}
    />
  );
}

function TeamPanel(props: {
  title: string;
  team: Team;
  picks: DraftPick[];
  heroLookup: Map<string, Hero>;
  heroByCode: Map<string, Hero>;
  language: AppLanguage;
  selectedBanCodes: Set<string>;
  currentDraftStep: CurrentDraftStep;
  emptySlotLabel: string;
  bannedLabel: string;
}) {
  const slots = Array.from({ length: MAX_TEAM_SIZE }, (_, index) => ({
    pick: props.picks[index],
    isBanProtected: index === BAN_PROTECTED_SLOT_INDEX,
    isCurrentStep:
      props.currentDraftStep?.phase === "pick" &&
      props.currentDraftStep.team === props.team &&
      props.currentDraftStep.slotIndex === index,
  }));

  return (
    <section className="team-column">
      <div className="panel-heading">
        <h2>{props.title}</h2>
      </div>

      <div className="pick-list">
        {slots.map((slot, index) => {
          const hero = slot.pick ? props.heroByCode.get(slot.pick.code) : null;
          const showBanMark = Boolean(slot.pick && props.selectedBanCodes.has(slot.pick.code));
          return (
            <div
              className={`pick-slot${slot.pick ? " filled" : ""}${slot.isBanProtected ? " ban-protected" : ""}${slot.isCurrentStep ? " current-step" : ""}`}
              key={slot.pick?.code ?? index}
              title={
                slot.pick
                  ? getHeroName(props.heroLookup, slot.pick.code, props.language)
                  : props.emptySlotLabel
              }
            >
              {hero ? (
                <div className={`pick-slot-hero${showBanMark ? " banned" : ""}`}>
                  <HeroAvatar
                    hero={hero}
                    displayName={getHeroName(props.heroLookup, slot.pick!.code, props.language)}
                    size="small"
                  />
                  {showBanMark && <PickBanOverlay label={props.bannedLabel} />}
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function DraftPanel(props: {
  userPrebans: DraftPick[];
  enemyPrebans: DraftPick[];
  userPicks: DraftPick[];
  enemyPicks: DraftPick[];
  heroLookup: Map<string, Hero>;
  heroByCode: Map<string, Hero>;
  language: AppLanguage;
  canUndo: boolean;
  onUndo: () => void;
  selectedBanCodes: Set<string>;
  currentDraftStep: CurrentDraftStep;
  labels: {
    draft: string;
    undo: string;
    preban: string;
    pick: string;
    ally: string;
    enemy: string;
    allyPrebanSlot: string;
    enemyPrebanSlot: string;
    emptySlot: string;
    banned: string;
    allyPrebanHeading: string;
  };
}) {
  return (
    <section className="draft-panel">
      <div className="panel-heading draft-heading">
        <div>
          <h2>{props.labels.draft}</h2>
        </div>
        <button type="button" className="panel-link-button" onClick={props.onUndo} disabled={!props.canUndo}>
          {props.labels.undo}
        </button>
      </div>
      <div className="preban-section">
        <span>{props.labels.preban}</span>
        <div className="preban-columns">
          <div className="preban-column">
            <strong>{props.labels.allyPrebanHeading}</strong>
            <div className="preban-slots">
              {Array.from({ length: 2 }, (_, index) => {
                const preban = props.userPrebans[index];
                const hero = preban ? props.heroByCode.get(preban.code) : null;
                const isCurrentStep =
                  props.currentDraftStep?.phase === "preban" &&
                  props.currentDraftStep.team === "user" &&
                  props.currentDraftStep.slotIndex === index;

                return (
                  <div
                    className={`pick-slot preban-slot${preban ? " filled" : ""}${isCurrentStep ? " current-step" : ""}`}
                    key={preban?.code ?? index}
                    title={
                      preban
                        ? getHeroName(props.heroLookup, preban.code, props.language)
                        : props.labels.allyPrebanSlot
                    }
                  >
                    {hero && (
                      <HeroAvatar
                        hero={hero}
                        displayName={getHeroName(props.heroLookup, preban.code, props.language)}
                        size="small"
                      />
                    )}
                  </div>
                );
              })}
            </div>
          </div>
          <div className="preban-column">
            <strong>{props.labels.enemy}</strong>
            <div className="preban-slots">
              {Array.from({ length: 2 }, (_, index) => {
                const preban = props.enemyPrebans[index];
                const hero = preban ? props.heroByCode.get(preban.code) : null;
                const isCurrentStep =
                  props.currentDraftStep?.phase === "preban" &&
                  props.currentDraftStep.team === "enemy" &&
                  props.currentDraftStep.slotIndex === index;

                return (
                  <div
                    className={`pick-slot preban-slot${preban ? " filled" : ""}${isCurrentStep ? " current-step" : ""}`}
                    key={preban?.code ?? index}
                    title={
                      preban
                        ? getHeroName(props.heroLookup, preban.code, props.language)
                        : props.labels.enemyPrebanSlot
                    }
                  >
                    {hero && (
                      <HeroAvatar
                        hero={hero}
                        displayName={getHeroName(props.heroLookup, preban.code, props.language)}
                        size="small"
                      />
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
      <span className="section-label">{props.labels.pick}</span>
      <div className="team-columns">
        <TeamPanel
          title={props.labels.ally}
          team="user"
          picks={props.userPicks}
          heroLookup={props.heroLookup}
          heroByCode={props.heroByCode}
          language={props.language}
          selectedBanCodes={props.selectedBanCodes}
          currentDraftStep={props.currentDraftStep}
          emptySlotLabel={props.labels.emptySlot}
          bannedLabel={props.labels.banned}
        />
        <TeamPanel
          title={props.labels.enemy}
          team="enemy"
          picks={props.enemyPicks}
          heroLookup={props.heroLookup}
          heroByCode={props.heroByCode}
          language={props.language}
          selectedBanCodes={props.selectedBanCodes}
          currentDraftStep={props.currentDraftStep}
          emptySlotLabel={props.labels.emptySlot}
          bannedLabel={props.labels.banned}
        />
      </div>
    </section>
  );
}

export default function App() {
  const [language, setLanguage] = useState<AppLanguage>(() => getStoredLanguage());
  const [heroes, setHeroes] = useState<Hero[]>([]);
  const [prebanPicks, setPrebanPicks] = useState<DraftPick[]>([]);
  const [draftPicks, setDraftPicks] = useState<DraftPick[]>([]);
  const [elementFilter, setElementFilter] = useState("all");
  const [roleFilter, setRoleFilter] = useState("all");
  const [searchText, setSearchText] = useState("");
  const [firstPickTeam, setFirstPickTeam] = useState<FirstPickTeam>("My Team");
  const [warfareRule, setWarfareRule] = useState<WarfareRule>("ANY");
  const [rememberPreban, setRememberPreban] = useState(false);
  const [prebanMemoryMode, setPrebanMemoryMode] = useState<PrebanMemoryMode>("shared");
  const [allyPrebanPresets, setAllyPrebanPresets] = useState<AllyPrebanPresets>(() => ({
    ...EMPTY_ALLY_PREBAN_PRESETS,
  }));
  const [isLoadingHeroes, setIsLoadingHeroes] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [aiRecommendation, setAiRecommendation] = useState<RecommendationResponse | null>(null);
  const [prebanSuggestionsCache, setPrebanSuggestionsCache] = useState<RecommendationResponse | null>(null);
  const [aiRecommendLoading, setAiRecommendLoading] = useState(false);
  const [prebanRecommendLoading, setPrebanRecommendLoading] = useState(false);
  const [aiRecommendError, setAiRecommendError] = useState<string | null>(null);
  const [selectedBanCode, setSelectedBanCode] = useState<string | null>(null);

  useEffect(() => {
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  }, [language]);

  function changeLanguage(nextLanguage: AppLanguage) {
    setLanguage(nextLanguage);
    storeLanguage(nextLanguage);
  }

  const displayHeroes = useMemo(() => localizeHeroes(heroes, language), [heroes, language]);

  const draftLabels = useMemo(
    () => ({
      draft: t(language, "draft"),
      undo: t(language, "undo"),
      preban: t(language, "preban"),
      pick: t(language, "pick"),
      ally: t(language, "ally"),
      enemy: t(language, "enemy"),
      allyPrebanSlot: t(language, "allyPrebanSlot"),
      enemyPrebanSlot: t(language, "enemyPrebanSlot"),
      emptySlot: t(language, "emptySlot"),
      banned: t(language, "banned"),
      allyPrebanHeading:
        rememberPreban && prebanMemoryMode === "split"
          ? `${t(language, "ally")} · ${
              firstPickTeam === "My Team"
                ? t(language, "allyFirstPrebans")
                : t(language, "enemyFirstPrebans")
            }`
          : t(language, "ally"),
    }),
    [firstPickTeam, language, prebanMemoryMode, rememberPreban],
  );

  const userPicks = useMemo(() => draftPicks.filter((pick) => pick.team === "user"), [draftPicks]);
  const enemyPicks = useMemo(() => draftPicks.filter((pick) => pick.team === "enemy"), [draftPicks]);
  const userPrebans = useMemo(() => prebanPicks.filter((pick) => pick.team === "user"), [prebanPicks]);
  const enemyPrebans = useMemo(() => prebanPicks.filter((pick) => pick.team === "enemy"), [prebanPicks]);

  const sortedHeroes = useMemo(
    () =>
      [...displayHeroes].sort(
        (a, b) =>
          (b.appearance_count ?? 0) - (a.appearance_count ?? 0) ||
          a.name.localeCompare(b.name, language === "zh" ? "zh-CN" : "en"),
      ),
    [displayHeroes, language],
  );

  const heroLookup = useMemo(() => new Map(heroes.map((hero) => [hero.code, hero])), [heroes]);
  const heroByCode = useMemo(() => new Map(displayHeroes.map((hero) => [hero.code, hero])), [displayHeroes]);
  const selectedBanSet = useMemo(
    () => new Set(selectedBanCode ? [selectedBanCode] : []),
    [selectedBanCode],
  );
  const draftComplete = userPicks.length >= MAX_TEAM_SIZE && enemyPicks.length >= MAX_TEAM_SIZE;
  const selectedCodes = useMemo(
    () => new Set([...prebanPicks, ...draftPicks].map((pick) => pick.code)),
    [draftPicks, prebanPicks],
  );
  const nextPickTeam = draftPicks.length < PICK_ORDER_PATTERN.length ? getPickTeam(firstPickTeam, draftPicks.length) : null;
  const canSelectHero = prebanPicks.length < MAX_PREBAN_SIZE || Boolean(nextPickTeam);
  const prebanOrder = PREBAN_ORDER;

  /** While filling enemy preban slots, hide "Picked" on heroes only ally pre-banned (dup allowed). */
  const pickingEnemyPreban =
    prebanPicks.length < MAX_PREBAN_SIZE && prebanOrder[prebanPicks.length] === "enemy";

  useEffect(() => {
    if (!draftComplete) {
      setSelectedBanCode(null);
      return;
    }
    setSelectedBanCode((prev) =>
      prev != null && isBanProtectedHero(prev, userPicks, enemyPicks) ? null : prev,
    );
  }, [draftComplete, userPicks, enemyPicks]);

  function selectBanTarget(code: string) {
    setSelectedBanCode((prev) => (prev === code ? null : code));
  }
  const heroesWithElement = useMemo(
    () => displayHeroes.filter((hero) => hero.element),
    [displayHeroes],
  );
  const elementOptions = useMemo(
    () =>
      sortByPredefinedOrder(
        displayHeroes.map((hero) => hero.element).filter(Boolean),
        ELEMENT_FILTER_ORDER,
      ).map((element) => ({
        value: element,
        label: localizeElement(element, language),
        iconUrl: displayHeroes.find((hero) => hero.element === element)?.element_icon_url ?? "",
      })),
    [displayHeroes, language],
  );
  const roleOptions = useMemo(
    () =>
      sortByPredefinedOrder(
        displayHeroes.map((hero) => hero.role).filter(Boolean),
        ROLE_FILTER_ORDER,
      ).map((role) => ({
        value: role,
        label: localizeRole(role, language),
        iconUrl: displayHeroes.find((hero) => hero.role === role)?.role_icon_url ?? "",
      })),
    [displayHeroes, language],
  );
  const filteredHeroes = useMemo(() => {
    return sortedHeroes.filter((hero) => {
      const matchesElement = elementFilter === "all" || hero.element === elementFilter;
      const matchesRole = roleFilter === "all" || hero.role === roleFilter;
      const matchesSearch = heroMatchesSearch(hero, searchText, language);
      return matchesElement && matchesRole && matchesSearch;
    });
  }, [elementFilter, language, roleFilter, searchText, sortedHeroes]);

  const inPrebanPhase = prebanPicks.length < MAX_PREBAN_SIZE;
  const currentPrebanSide: PrebanSide | null = inPrebanPhase ? prebanOrder[prebanPicks.length] : null;
  const currentDraftStep = useMemo((): CurrentDraftStep => {
    if (inPrebanPhase && currentPrebanSide) {
      const team: Team = currentPrebanSide === "user" ? "user" : "enemy";
      const slotIndex = team === "user" ? userPrebans.length : enemyPrebans.length;
      return { phase: "preban", team, slotIndex };
    }
    if (nextPickTeam) {
      const slotIndex = nextPickTeam === "user" ? userPicks.length : enemyPicks.length;
      return { phase: "pick", team: nextPickTeam, slotIndex };
    }
    return null;
  }, [
    currentPrebanSide,
    enemyPrebans.length,
    enemyPicks.length,
    inPrebanPhase,
    nextPickTeam,
    userPrebans.length,
    userPicks.length,
  ]);

  /** Hide picked heroes in picker; during preban only hide current side's locks. */
  const heroPickerList = useMemo(() => {
    if (inPrebanPhase && currentPrebanSide) {
      const pickedOnCurrentSide = new Set(
        prebanPicks.filter((pick) => pick.team === currentPrebanSide).map((pick) => pick.code),
      );
      return filteredHeroes.filter((hero) => !pickedOnCurrentSide.has(hero.code));
    }
    if (prebanPicks.length >= MAX_PREBAN_SIZE) {
      return filteredHeroes.filter((hero) => !selectedCodes.has(hero.code));
    }
    return filteredHeroes;
  }, [currentPrebanSide, filteredHeroes, inPrebanPhase, prebanPicks, selectedCodes]);

  useEffect(() => {
    fetchHeroes()
      .then((nextHeroes) => {
        setHeroes(nextHeroes);
        setError(null);
      })
      .catch((nextError) => {
        setError(nextError instanceof Error ? nextError.message : t(language, "loadHeroesFailed"));
      })
      .finally(() => setIsLoadingHeroes(false));
  }, [language]);

  const userPickCodesForAi = useMemo(() => userPicks.map((pick) => pick.code), [userPicks]);
  const enemyPickCodesForAi = useMemo(() => enemyPicks.map((pick) => pick.code), [enemyPicks]);
  const userPrebanCodesForAi = useMemo(() => userPrebans.map((pick) => pick.code), [userPrebans]);
  const enemyPrebanCodesForAi = useMemo(() => enemyPrebans.map((pick) => pick.code), [enemyPrebans]);

  const prebanRecommendation = useMemo(() => {
    if (!prebanSuggestionsCache || !currentPrebanSide) {
      return null;
    }

    const originalCodes = prebanSuggestionsCache.top_10_heroes;
    const originalCodeSet = new Set(originalCodes);
    const pickedFromSuggestions = new Set(
      prebanPicks
        .filter((pick) => pick.team === currentPrebanSide)
        .map((pick) => pick.code)
        .filter((code) => originalCodeSet.has(code)),
    );

    const recommendations = (prebanSuggestionsCache.recommendations ?? []).filter(
      (item) => !pickedFromSuggestions.has(item.hero_id),
    );

    return {
      ...prebanSuggestionsCache,
      recommendations,
      top_10_heroes: recommendations.map((item) => item.hero_id),
      top_10_rates: recommendations.map((item) => item.normalized_preban_rate * 100.0),
    };
  }, [currentPrebanSide, prebanPicks, prebanSuggestionsCache]);

  const pickRecommendation = useMemo(() => {
    if (!aiRecommendation || aiRecommendation.phase === "ban") {
      return aiRecommendation;
    }

    const filteredEntries = aiRecommendation.top_10_heroes
      .map((code, index) => ({
        code,
        rate: aiRecommendation.top_10_rates?.[index],
      }))
      .filter((entry) => !selectedCodes.has(entry.code));

    return {
      ...aiRecommendation,
      top_10_heroes: filteredEntries.map((entry) => entry.code),
      top_10_rates: aiRecommendation.top_10_rates
        ? filteredEntries.map((entry) => entry.rate ?? Number.NaN)
        : undefined,
    };
  }, [aiRecommendation, selectedCodes]);

  const activeRecommendation = inPrebanPhase ? prebanRecommendation : pickRecommendation;
  const activeRecommendLoading = inPrebanPhase ? prebanRecommendLoading : aiRecommendLoading;
  const showPickRecommendLoading = !inPrebanPhase && aiRecommendLoading;
  const showPrebanRecommendLoading =
    inPrebanPhase &&
    prebanRecommendLoading &&
    (!prebanRecommendation || prebanRecommendation.top_10_heroes.length === 0);
  const banSuggestionsLocked =
    activeRecommendation?.phase === "ban" && selectedBanCode != null;

  useEffect(() => {
    if (isLoadingHeroes || heroes.length === 0 || !inPrebanPhase || !currentPrebanSide) {
      return;
    }

    let cancelled = false;
    setPrebanSuggestionsCache(null);
    setPrebanRecommendLoading(true);

    fetchPrebanRecommendation({
      excludedHeroes: [],
      topK: PREBAN_SUGGESTION_POOL_SIZE,
      prebanSide: currentPrebanSide,
      firstPickTeam,
    })
      .then((data) => {
        if (!cancelled) {
          setPrebanSuggestionsCache(data);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setPrebanSuggestionsCache(null);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setPrebanRecommendLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [currentPrebanSide, firstPickTeam, heroes.length, inPrebanPhase, isLoadingHeroes]);

  useEffect(() => {
    if (isLoadingHeroes || heroes.length === 0 || inPrebanPhase) {
      return;
    }

    setAiRecommendLoading(true);
    setAiRecommendError(null);

    const timer = window.setTimeout(() => {
      fetchRecommendation({
        userPicks: userPickCodesForAi,
        enemyPicks: enemyPickCodesForAi,
        allyPreban: userPrebanCodesForAi,
        enemyPreban: enemyPrebanCodesForAi,
        firstPickTeam,
        warfareRules: warfareRule,
      })
        .then((data) => {
          setAiRecommendation(data);
        })
        .catch((nextError) => {
          setAiRecommendError(nextError instanceof Error ? nextError.message : t(language, "loadPickSuggestionsFailed"));
        })
        .finally(() => {
          setAiRecommendLoading(false);
        });
    }, 380);

    return () => window.clearTimeout(timer);
  }, [
    enemyPickCodesForAi,
    enemyPrebanCodesForAi,
    firstPickTeam,
    heroes.length,
    inPrebanPhase,
    isLoadingHeroes,
    userPickCodesForAi,
    userPrebanCodesForAi,
    warfareRule,
    language,
  ]);

  function addPick(code: string) {
    if (!canSelectHero || isHeroUnavailableForNextPick(code, prebanPicks, selectedCodes, prebanOrder)) {
      return;
    }

    if (prebanPicks.length < MAX_PREBAN_SIZE) {
      setPrebanPicks((current) => [...current, { team: prebanOrder[current.length], code }]);
      return;
    }

    if (!nextPickTeam) {
      return;
    }

    setDraftPicks((current) => [...current, { team: nextPickTeam, code }]);
  }

  function undoLastPick() {
    if (selectedBanCode != null) {
      setSelectedBanCode(null);
      return;
    }

    if (draftPicks.length > 0) {
      setDraftPicks((current) => current.slice(0, -1));
      return;
    }

    setPrebanPicks((current) => current.slice(0, -1));
  }

  function resetDraft(nextFirstPickTeam: FirstPickTeam) {
    if (rememberPreban && prebanMemoryMode === "split") {
      const currentUserPrebans = extractUserPrebans(prebanPicks);
      const updatedPresets: AllyPrebanPresets = {
        ...allyPrebanPresets,
        [firstPickTeam]: currentUserPrebans,
      };
      setAllyPrebanPresets(updatedPresets);
      setPrebanPicks(prebanPicksFromUserPresets(updatedPresets[nextFirstPickTeam]));
    } else if (rememberPreban) {
      setPrebanPicks((current) => current.filter((pick) => pick.team === "user"));
    } else {
      setPrebanPicks([]);
    }

    setFirstPickTeam(nextFirstPickTeam);
    setDraftPicks([]);
    setPrebanSuggestionsCache(null);
  }

  function toggleRememberPreban() {
    setRememberPreban((current) => {
      const next = !current;
      if (!next) {
        setPrebanMemoryMode("shared");
      }
      return next;
    });
  }

  function choosePrebanMemoryMode(nextMode: PrebanMemoryMode) {
    if (nextMode === "split") {
      const currentUserPrebans = extractUserPrebans(prebanPicks);
      setAllyPrebanPresets((presets) => ({
        ...presets,
        [firstPickTeam]: currentUserPrebans,
      }));
    }
    setPrebanMemoryMode(nextMode);
  }

  return (
    <main className="app-shell">
      {error && <div className="error-banner">{error}</div>}

      <div className="draft-grid">
        <section className="control-panel">
          <div className="settings-heading">
            <h2>{t(language, "settings")}</h2>
            <div className="language-controls" role="group" aria-label={t(language, "language")}>
              <span className="language-label">{t(language, "language")}</span>
              <button
                type="button"
                className={language === "zh" ? "active" : ""}
                onClick={() => changeLanguage("zh")}
              >
                {t(language, "languageZh")}
              </button>
              <button
                type="button"
                className={language === "en" ? "active" : ""}
                onClick={() => changeLanguage("en")}
              >
                {t(language, "languageEn")}
              </button>
            </div>
          </div>
          <div className="first-pick-controls">
            <button
              type="button"
              className={firstPickTeam === "My Team" ? "active" : ""}
              onClick={() => {
                resetDraft("My Team");
              }}
            >
              {t(language, "allyFirst")}
            </button>
            <button
              type="button"
              className={firstPickTeam === "Enemy Team" ? "active" : ""}
              onClick={() => {
                resetDraft("Enemy Team");
              }}
            >
              {t(language, "enemyFirst")}
            </button>
            <div className="preban-settings">
              <button
                type="button"
                className={`preban-button${rememberPreban ? " active" : ""}`}
                onClick={toggleRememberPreban}
              >
                {t(language, "rememberPreban")}
              </button>
              {rememberPreban && (
                <div className="preban-memory-options" role="group" aria-label={t(language, "rememberPreban")}>
                  <button
                    type="button"
                    className={prebanMemoryMode === "shared" ? "active" : ""}
                    onClick={() => choosePrebanMemoryMode("shared")}
                  >
                    {t(language, "sharedPrebans")}
                  </button>
                  <button
                    type="button"
                    className={prebanMemoryMode === "split" ? "active" : ""}
                    onClick={() => choosePrebanMemoryMode("split")}
                  >
                    {t(language, "splitPrebanByFirstPick")}
                  </button>
                </div>
              )}
            </div>
          </div>
          <div className="warfare-rule-section">
            <span className="warfare-rule-label">{t(language, "warfareRules")}</span>
            <div className="warfare-rule-controls" role="group" aria-label={t(language, "warfareRules")}>
              {WARFARE_RULE_OPTIONS.map((option) => (
                <button
                  key={option}
                  type="button"
                  className={warfareRule === option ? "active" : ""}
                  onClick={() => setWarfareRule(option)}
                >
                  {t(language, `warfareRule${option}` as MessageKey)}
                </button>
              ))}
            </div>
          </div>
        </section>

        <DraftPanel
          userPrebans={userPrebans}
          enemyPrebans={enemyPrebans}
          userPicks={userPicks}
          enemyPicks={enemyPicks}
          heroLookup={heroLookup}
          heroByCode={heroByCode}
          language={language}
          canUndo={selectedBanCode != null || prebanPicks.length > 0 || draftPicks.length > 0}
          onUndo={undoLastPick}
          selectedBanCodes={selectedBanSet}
          currentDraftStep={currentDraftStep}
          labels={draftLabels}
        />

        <div className="picker-columns">
          <section className="hero-picker-panel">
          <div className="panel-heading">
            <div>
              <h2>{t(language, "heroPicker")}</h2>
              <span>
                {heroPickerList.length} {t(language, "shown")}
                {heroesWithElement.length !== displayHeroes.length
                  ? ` · ${heroesWithElement.length}/${displayHeroes.length} ${t(language, "tagged")}`
                  : ""}
              </span>
            </div>
          </div>

          <div className="hero-picker-body">
            <div className="filters">
            <input
              value={searchText}
              onChange={(event) => setSearchText(event.target.value)}
              placeholder={t(language, "searchHero")}
            />
            <div className="icon-filter-group" aria-label={t(language, "elementFilters")}>
              <button
                type="button"
                className={elementFilter === "all" ? "active" : ""}
                onClick={() => setElementFilter("all")}
              >
                {t(language, "all")}
              </button>
              {elementOptions.map((element) => (
                <button
                  type="button"
                  key={element.value}
                  className={elementFilter === element.value ? "active" : ""}
                  onClick={() =>
                    setElementFilter(elementFilter === element.value ? "all" : element.value)
                  }
                  title={element.label}
                >
                  {element.iconUrl ? <img src={element.iconUrl} alt={element.label} /> : element.label}
                </button>
              ))}
            </div>
            <div className="icon-filter-group" aria-label={t(language, "roleFilters")}>
              <button
                type="button"
                className={roleFilter === "all" ? "active" : ""}
                onClick={() => setRoleFilter("all")}
              >
                {t(language, "all")}
              </button>
              {roleOptions.map((role) => (
                <button
                  type="button"
                  key={role.value}
                  className={roleFilter === role.value ? "active" : ""}
                  onClick={() => setRoleFilter(roleFilter === role.value ? "all" : role.value)}
                  title={role.label}
                >
                  {role.iconUrl ? <img src={role.iconUrl} alt={role.label} /> : role.label}
                </button>
              ))}
            </div>
          </div>

          <div className="hero-grid">
            {heroPickerList.map((hero) => {
              const heroBlocked = isHeroUnavailableForNextPick(
                hero.code,
                prebanPicks,
                selectedCodes,
                prebanOrder,
              );
              return (
                <button
                  className="hero-card"
                  key={hero.code}
                  type="button"
                  onClick={() => addPick(hero.code)}
                  disabled={heroBlocked || !canSelectHero}
                  title={getHeroName(heroLookup, hero.code, language)}
                >
                  <HeroAvatar
                    hero={hero}
                    displayName={getHeroName(heroLookup, hero.code, language)}
                  />
                </button>
              );
            })}
          </div>
          </div>
        </section>

          <section
            className="ai-recommend-panel"
            aria-label={
              banSuggestionsLocked
                ? t(language, "banPhaseCompleted")
                : activeRecommendation?.phase === "ban"
                  ? t(language, "banSuggestions")
                  : activeRecommendation?.phase === "preban"
                    ? t(language, "prebanSuggestions")
                    : t(language, "pickSuggestions")
            }
          >
            <div className="panel-heading">
              <div>
                <h2>
                  {activeRecommendation?.phase === "ban"
                    ? t(language, "banSuggestions")
                    : activeRecommendation?.phase === "preban"
                      ? t(language, "prebanSuggestions")
                      : t(language, "pickSuggestions")}
                </h2>
              </div>
            </div>
            {aiRecommendError && <p className="ai-recommend-error">{aiRecommendError}</p>}
            <div className="ai-recommend-list">
              {(showPickRecommendLoading || showPrebanRecommendLoading) && (
                <p className="empty-state">{t(language, "loadingSuggestions")}</p>
              )}
              {!activeRecommendLoading && banSuggestionsLocked && (
                <p className="empty-state ai-recommend-completed">{t(language, "completed")}</p>
              )}
              {!activeRecommendLoading &&
                !banSuggestionsLocked &&
                activeRecommendation &&
                activeRecommendation.top_10_heroes.length === 0 && (
                  <p className="empty-state">
                    {activeRecommendation.phase === "preban"
                      ? t(language, "noPrebanData")
                      : activeRecommendation.phase === "ban"
                        ? t(language, "noBanData")
                        : t(language, "draftComplete")}
                  </p>
                )}
              {!banSuggestionsLocked &&
                !showPickRecommendLoading &&
                activeRecommendation?.top_10_heroes.map((code, index) => {
                  const hero = heroByCode.get(code);
                  const blocked = isHeroUnavailableForNextPick(
                    code,
                    prebanPicks,
                    selectedCodes,
                    prebanOrder,
                  );
                  const banPhase = activeRecommendation.phase === "ban";
                  const prebanPhase = activeRecommendation.phase === "preban";
                  const banProtected = banPhase && isBanProtectedHero(code, userPicks, enemyPicks);
                  const banChosen = banPhase && selectedBanCode === code;
                  const disabled = banPhase
                    ? !hero || banProtected
                    : !canSelectHero || blocked || !hero;
                  const rate = activeRecommendation.top_10_rates?.[index];
                  const rateLabel =
                    rate != null && !Number.isNaN(rate)
                      ? `${rate.toFixed(1)}%`
                      : "—";
                  const hidePrebanPct = prebanPhase && pickingEnemyPreban;
                  const cardTitle = getHeroName(heroLookup, code, language);
                  return (
                    <button
                      type="button"
                      key={`${code}-${index}`}
                      className={`ai-recommend-card${banChosen ? " ban-suggestion-selected" : ""}`}
                      aria-pressed={banPhase ? banChosen : undefined}
                      disabled={disabled}
                      onClick={() => {
                        if (banPhase) {
                          if (!banProtected && hero) selectBanTarget(code);
                          return;
                        }
                        if (hero) addPick(code);
                      }}
                      title={cardTitle}
                    >
                      {hero ? (
                        <HeroAvatar
                          hero={hero}
                          displayName={cardTitle}
                          size="small"
                        />
                      ) : (
                        <span className="avatar-fallback small">?</span>
                      )}
                      {!hidePrebanPct ? <span className="ai-recommend-pct">{rateLabel}</span> : null}
                    </button>
                  );
                })}
            </div>
          </section>
        </div>

      </div>
    </main>
  );
}
