import { useEffect, useMemo, useState } from "react";
import { fetchHeroes, fetchPrebanRecommendation, fetchRecommendation, type PrebanSide } from "./api";
import { CompactMenu } from "./components/CompactMenu";
import { DraftPanel } from "./components/DraftPanel";
import { HeroAvatar } from "./components/HeroAvatar";
import {
  banPriorityLabelKey,
  ELEMENT_FILTER_ORDER,
  EMPTY_ALLY_PREBAN_PRESETS,
  getHeroName,
  getPickTeam,
  isBanProtectedHero,
  isHeroUnavailableForNextPick,
  MAX_PREBAN_SIZE,
  MAX_TEAM_SIZE,
  mergePrebanPicks,
  PICK_ORDER_PATTERN,
  PREBAN_ORDER,
  PREBAN_SUGGESTION_DISPLAY_SIZE,
  PREBAN_SUGGESTION_POOL_SIZE,
  picksByTeam,
  prebanPicksFromUserPresets,
  ROLE_FILTER_ORDER,
  sortByPredefinedOrder,
  WARFARE_RULE_OPTIONS,
  type AllyPrebanPresets,
  type CurrentDraftStep,
  type DraftPick,
  type PrebanMemoryMode,
  type Team,
} from "./draftLogic";
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
  const [languageMenuOpen, setLanguageMenuOpen] = useState(false);
  const [firstPickMenuOpen, setFirstPickMenuOpen] = useState(false);
  const [warfareMenuOpen, setWarfareMenuOpen] = useState(false);
  const [elementMenuOpen, setElementMenuOpen] = useState(false);
  const [roleMenuOpen, setRoleMenuOpen] = useState(false);
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
      resetDraft: t(language, "resetDraft"),
      preban: t(language, "preban"),
      pick: t(language, "pick"),
      ally: t(language, "ally"),
      enemy: t(language, "enemy"),
      allyPrebanSlot: t(language, "allyPrebanSlot"),
      enemyPrebanSlot: t(language, "enemyPrebanSlot"),
      emptySlot: t(language, "emptySlot"),
      banned: t(language, "banned"),
      allyPrebanFirstPickSuffix:
        rememberPreban && prebanMemoryMode === "split"
          ? ` · ${
              firstPickTeam === "My Team"
                ? t(language, "allyFirstPrebans")
                : t(language, "enemyFirstPrebans")
            }`
          : null,
    }),
    [firstPickTeam, language, prebanMemoryMode, rememberPreban],
  );

  const userPicks = useMemo(() => picksByTeam(draftPicks, "user"), [draftPicks]);
  const enemyPicks = useMemo(() => picksByTeam(draftPicks, "enemy"), [draftPicks]);
  const userPrebans = useMemo(() => picksByTeam(prebanPicks, "user"), [prebanPicks]);
  const enemyPrebans = useMemo(() => picksByTeam(prebanPicks, "enemy"), [prebanPicks]);

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

    const recommendations = (prebanSuggestionsCache.recommendations ?? [])
      .filter((item) => !pickedFromSuggestions.has(item.hero_id))
      .slice(0, PREBAN_SUGGESTION_DISPLAY_SIZE);

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

  function applyDraftStateChange(
    nextFirstPickTeam: FirstPickTeam,
    options: { preserveEnemyPrebans: boolean },
  ) {
    const enemyPrebans = options.preserveEnemyPrebans ? picksByTeam(prebanPicks, "enemy") : [];

    if (rememberPreban && prebanMemoryMode === "split") {
      const currentUserPrebans = picksByTeam(prebanPicks, "user");
      const updatedPresets: AllyPrebanPresets = {
        ...allyPrebanPresets,
        [firstPickTeam]: currentUserPrebans,
      };
      setAllyPrebanPresets(updatedPresets);
      const allyPrebans = prebanPicksFromUserPresets(updatedPresets[nextFirstPickTeam]);
      setPrebanPicks(mergePrebanPicks(allyPrebans, enemyPrebans));
    } else if (rememberPreban) {
      const allyPrebans = picksByTeam(prebanPicks, "user");
      setPrebanPicks(mergePrebanPicks(allyPrebans, enemyPrebans));
    } else {
      setPrebanPicks(enemyPrebans);
    }

    setFirstPickTeam(nextFirstPickTeam);
    setDraftPicks([]);
    setSelectedBanCode(null);
    setPrebanSuggestionsCache(null);
  }

  function switchFirstPickTeam(nextFirstPickTeam: FirstPickTeam) {
    if (nextFirstPickTeam === firstPickTeam) {
      return;
    }
    applyDraftStateChange(nextFirstPickTeam, { preserveEnemyPrebans: true });
  }

  function resetDraft(nextFirstPickTeam: FirstPickTeam) {
    applyDraftStateChange(nextFirstPickTeam, { preserveEnemyPrebans: false });
  }

  function resetCurrentDraft() {
    setSelectedBanCode(null);
    resetDraft(firstPickTeam);
  }

  function handleFirstPickTeamSelect(nextFirstPickTeam: FirstPickTeam) {
    if (nextFirstPickTeam === firstPickTeam) {
      resetCurrentDraft();
      return;
    }
    switchFirstPickTeam(nextFirstPickTeam);
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

  function choosePrebanMemoryMode(nextMode: PrebanMemoryMode) {
    if (nextMode === "split") {
      const currentUserPrebans = picksByTeam(prebanPicks, "user");
      setAllyPrebanPresets((presets) => ({
        ...presets,
        [firstPickTeam]: currentUserPrebans,
      }));
    }
    setPrebanMemoryMode(nextMode);
  }

  function toggleMobileMenu(menu: "language" | "firstPick" | "warfare" | "element" | "role") {
    setLanguageMenuOpen((open) => (menu === "language" ? !open : false));
    setFirstPickMenuOpen((open) => (menu === "firstPick" ? !open : false));
    setWarfareMenuOpen((open) => (menu === "warfare" ? !open : false));
    setElementMenuOpen((open) => (menu === "element" ? !open : false));
    setRoleMenuOpen((open) => (menu === "role" ? !open : false));
  }

  return (
    <main className="app-shell">
      {error && <div className="error-banner">{error}</div>}

      <div className="draft-grid">
        <section className="control-panel">
          <div className="panel-heading settings-heading">
            <h2 className="panel-title">{t(language, "settings")}</h2>
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
              <div className="language-menu-mobile">
                <CompactMenu
                  ariaLabel={t(language, "language")}
                  value={language}
                  options={[
                    { value: "zh", label: t(language, "languageZh") },
                    { value: "en", label: t(language, "languageEn") },
                  ]}
                  open={languageMenuOpen}
                  onToggle={() => toggleMobileMenu("language")}
                  onSelect={(nextLanguage) => {
                    changeLanguage(nextLanguage);
                    setLanguageMenuOpen(false);
                  }}
                />
              </div>
            </div>
          </div>
          <div className="first-pick-controls">
            <button
              type="button"
              className={firstPickTeam === "My Team" ? "active" : ""}
              onClick={() => {
                handleFirstPickTeamSelect("My Team");
              }}
            >
              {t(language, "allyFirst")}
            </button>
            <button
              type="button"
              className={firstPickTeam === "Enemy Team" ? "active" : ""}
              onClick={() => {
                handleFirstPickTeamSelect("Enemy Team");
              }}
            >
              {t(language, "enemyFirst")}
            </button>
            <div className="first-pick-menu-mobile">
              <span>{t(language, "firstPick")}</span>
              <div className="first-pick-menu-control">
                <CompactMenu
                  ariaLabel={t(language, "firstPick")}
                  value={firstPickTeam}
                  options={[
                    { value: "My Team", label: t(language, "ally") },
                    { value: "Enemy Team", label: t(language, "enemy") },
                  ]}
                  open={firstPickMenuOpen}
                  onToggle={() => toggleMobileMenu("firstPick")}
                  onSelect={(nextFirstPickTeam) => {
                    handleFirstPickTeamSelect(nextFirstPickTeam);
                    setFirstPickMenuOpen(false);
                  }}
                />
              </div>
            </div>
            <div className="preban-settings">
              <div className="preban-toggle-row desktop-preban-toggle">
                <span className="preban-toggle-label">{t(language, "rememberPreban")}</span>
                <div className="preban-toggle-options" role="group" aria-label={t(language, "rememberPreban")}>
                  <button
                    type="button"
                    className={rememberPreban ? "active" : ""}
                    onClick={() => setRememberPreban(true)}
                  >
                    {t(language, "toggleOn")}
                  </button>
                  <button
                    type="button"
                    className={!rememberPreban ? "active" : ""}
                    onClick={() => {
                      setRememberPreban(false);
                      setPrebanMemoryMode("shared");
                    }}
                  >
                    {t(language, "toggleOff")}
                  </button>
                </div>
              </div>
              <div className="preban-toggle-row mobile-preban-toggle">
                <span className="preban-toggle-label">{t(language, "rememberPreban")}</span>
                <button
                  type="button"
                  role="switch"
                  aria-checked={rememberPreban}
                  aria-label={t(language, "rememberPreban")}
                  className={`ios-switch${rememberPreban ? " on" : ""}`}
                  onClick={() => {
                    setRememberPreban((current) => {
                      const next = !current;
                      if (!next) {
                        setPrebanMemoryMode("shared");
                      }
                      return next;
                    });
                  }}
                >
                  <span className="ios-switch-thumb" aria-hidden="true" />
                </button>
              </div>
              <div
                className={`preban-memory-options${rememberPreban ? "" : " is-hidden"}`}
                role="group"
                aria-label={t(language, "rememberPreban")}
                aria-hidden={!rememberPreban}
              >
                <button
                  type="button"
                  className={prebanMemoryMode === "shared" ? "active" : ""}
                  disabled={!rememberPreban}
                  onClick={() => choosePrebanMemoryMode("shared")}
                >
                  {t(language, "sharedPrebans")}
                </button>
                <button
                  type="button"
                  className={prebanMemoryMode === "split" ? "active" : ""}
                  disabled={!rememberPreban}
                  onClick={() => choosePrebanMemoryMode("split")}
                >
                  {t(language, "splitPrebanByFirstPick")}
                </button>
              </div>
            </div>
          </div>
          <div className="warfare-rule-section">
            <div className="warfare-rule-label-row">
              <span className="warfare-rule-label">{t(language, "warfareRules")}</span>
              <span
                className="help-tooltip"
                tabIndex={0}
                aria-label={t(language, "warfareRulesHelp")}
              >
                <span className="help-tooltip-icon" aria-hidden="true">
                  ?
                </span>
                <span className="help-tooltip-content" role="tooltip">
                  {t(language, "warfareRulesHelp")}
                </span>
              </span>
            </div>
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
            <div className="warfare-rule-mobile">
              <CompactMenu
                ariaLabel={t(language, "warfareRules")}
                value={warfareRule}
                options={WARFARE_RULE_OPTIONS.map((option) => ({
                  value: option,
                  label: t(language, `warfareRule${option}` as MessageKey),
                }))}
                open={warfareMenuOpen}
                onToggle={() => toggleMobileMenu("warfare")}
                onSelect={(nextWarfareRule) => {
                  setWarfareRule(nextWarfareRule);
                  setWarfareMenuOpen(false);
                }}
              />
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
          canReset={selectedBanCode != null || prebanPicks.length > 0 || draftPicks.length > 0}
          onReset={resetCurrentDraft}
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
              <h2 className="panel-title">{t(language, "heroPicker")}</h2>
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
            <div className="icon-filter-group filter-group-desktop" aria-label={t(language, "elementFilters")}>
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
            <div className="icon-filter-group filter-group-desktop" aria-label={t(language, "roleFilters")}>
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
            <div className="filter-menu-row">
              <div className="filter-menu-mobile">
                <CompactMenu
                  ariaLabel={t(language, "elementFilters")}
                  value={elementFilter}
                  options={[
                    { value: "all", label: t(language, "all") },
                    ...elementOptions.map((element) => ({
                      value: element.value,
                      label: element.label,
                      iconUrl: element.iconUrl,
                    })),
                  ]}
                  iconOnly
                  open={elementMenuOpen}
                  onToggle={() => toggleMobileMenu("element")}
                  onSelect={(nextElement) => {
                    setElementFilter(nextElement);
                    setElementMenuOpen(false);
                  }}
                />
              </div>
              <div className="filter-menu-mobile">
                <CompactMenu
                  ariaLabel={t(language, "roleFilters")}
                  value={roleFilter}
                  options={[
                    { value: "all", label: t(language, "all") },
                    ...roleOptions.map((role) => ({
                      value: role.value,
                      label: role.label,
                      iconUrl: role.iconUrl,
                    })),
                  ]}
                  iconOnly
                  open={roleMenuOpen}
                  onToggle={() => toggleMobileMenu("role")}
                  onSelect={(nextRole) => {
                    setRoleFilter(nextRole);
                    setRoleMenuOpen(false);
                  }}
                />
              </div>
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
                <h2 className="panel-title">
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
                      {hidePrebanPct ? (
                        <span className="ai-recommend-meta is-reserved" aria-hidden="true">
                          <span className="ai-recommend-pct">{rateLabel}</span>
                        </span>
                      ) : (
                        <span className="ai-recommend-meta">
                          <span className="ai-recommend-pct">{rateLabel}</span>
                          {banPhase ? (
                            <span className={`ban-priority-label rank-${Math.min(index + 1, 4)}`}>
                              {t(language, banPriorityLabelKey(index))}
                            </span>
                          ) : null}
                        </span>
                      )}
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
