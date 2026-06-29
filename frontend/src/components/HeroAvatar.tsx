import { useState } from "react";
import type { Hero } from "../types";

export function HeroAvatar(props: { hero: Hero; displayName: string; size?: "small" | "large" }) {
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
      onError={() => setImageFailed(true)}
    />
  );
}
