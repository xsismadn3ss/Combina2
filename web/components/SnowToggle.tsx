"use client";

import { TreePine } from "lucide-react";
import { cn } from "@/lib/utils";
import { useSnow } from "./SnowProvider";

export function SnowToggle() {
  const { enabled, setEnabled } = useSnow();

  return (
    <span className="inline-flex items-center gap-2">
      <TreePine className="size-4 text-muted-foreground" aria-hidden />
      <button
        type="button"
        role="switch"
        aria-checked={enabled}
        aria-label="Nieve"
        title="Activar o desactivar la nieve"
        onClick={() => setEnabled(!enabled)}
        className={cn(
          "relative h-6 w-11 shrink-0 rounded-full transition-colors",
          enabled ? "bg-primary" : "bg-muted"
        )}
      >
        <span
          aria-hidden
          className={cn(
            "absolute top-0.5 size-5 rounded-full bg-background shadow transition-all",
            enabled ? "left-[22px]" : "left-0.5"
          )}
        />
      </button>
    </span>
  );
}
