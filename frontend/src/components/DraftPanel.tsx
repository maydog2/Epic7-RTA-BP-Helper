import { getHeroName, type CurrentDraftStep, type DraftPick } from "../draftLogic";
import type { AppLanguage } from "../i18n";
import type { Hero } from "../types";
import { HeroAvatar } from "./HeroAvatar";
import { MaterialSymbol } from "./MaterialSymbol";
import { TeamPanel } from "./TeamPanel";

export type DraftPanelLabels = {
  draft: string;
  undo: string;
  resetDraft: string;
  preban: string;
  pick: string;
  ally: string;
  enemy: string;
  allyPrebanSlot: string;
  enemyPrebanSlot: string;
  emptySlot: string;
  banned: string;
  allyPrebanFirstPickSuffix: string | null;
};

export function DraftPanel(props: {
  userPrebans: DraftPick[];
  enemyPrebans: DraftPick[];
  userPicks: DraftPick[];
  enemyPicks: DraftPick[];
  heroLookup: Map<string, Hero>;
  heroByCode: Map<string, Hero>;
  language: AppLanguage;
  canUndo: boolean;
  onUndo: () => void;
  canReset: boolean;
  onReset: () => void;
  selectedBanCodes: Set<string>;
  currentDraftStep: CurrentDraftStep;
  labels: DraftPanelLabels;
}) {
  return (
    <section className="draft-panel">
      <div className="panel-heading draft-heading">
        <div className="draft-heading-start">
          <h2 className="panel-title">{props.labels.draft}</h2>
          <button
            type="button"
            className="panel-link-button draft-reset-button"
            onClick={props.onReset}
            disabled={!props.canReset}
            aria-label={props.labels.resetDraft}
            title={props.labels.resetDraft}
          >
            <MaterialSymbol name="refresh" className="draft-action-icon" />
            <span className="draft-action-label">{props.labels.resetDraft}</span>
          </button>
        </div>
        <button
          type="button"
          className="panel-link-button draft-undo-button"
          onClick={props.onUndo}
          disabled={!props.canUndo}
          aria-label={props.labels.undo}
          title={props.labels.undo}
        >
          <MaterialSymbol name="undo" className="draft-action-icon" />
          <span className="draft-action-label">{props.labels.undo}</span>
        </button>
      </div>
      <div className="preban-section">
        <span>{props.labels.preban}</span>
        <div className="preban-columns">
          <div className="preban-column">
            <strong>
              {props.labels.ally}
              {props.labels.allyPrebanFirstPickSuffix ? (
                <span className="preban-ally-first-pick-suffix">
                  {props.labels.allyPrebanFirstPickSuffix}
                </span>
              ) : null}
            </strong>
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
