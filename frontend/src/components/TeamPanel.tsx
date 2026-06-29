import {
  BAN_PROTECTED_SLOT_INDEX,
  getHeroName,
  MAX_TEAM_SIZE,
  type CurrentDraftStep,
  type DraftPick,
  type Team,
} from "../draftLogic";
import type { AppLanguage } from "../i18n";
import type { Hero } from "../types";
import { HeroAvatar } from "./HeroAvatar";

export function TeamPanel(props: {
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
