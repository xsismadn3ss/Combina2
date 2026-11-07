"use client";

import Snowfall from "react-snowfall";
import { useTheme } from "next-themes";
import { useSnow } from "./SnowProvider";

export function SnowEffect() {
  const { enabled } = useSnow();
  const { resolvedTheme } = useTheme();

  if (!enabled) return null;

  // #dee4fd (default) is invisible on light backgrounds, so use a
  // stronger blue when the light theme is active.
  const color = resolvedTheme === "light" ? "#7aa5f0" : "#dee4fd";

  return (
    <Snowfall
      color={color}
      snowflakeCount={120}
      style={{
        position: "fixed",
        width: "100vw",
        height: "100vh",
        pointerEvents: "none",
        zIndex: 50,
      }}
    />
  );
}
