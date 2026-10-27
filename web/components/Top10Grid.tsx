"use client";
import { Check } from "lucide-react";
import type { Color } from "@/lib/api/types";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { cn } from "cn";

type Props = {
  colors: Color[];
  selected: string[];
  onToggle: (hex: string) => void;
  maxSelected?: number;
};

export function Top10Grid({ colors, selected, onToggle, maxSelected = 4 }: Props) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Colores de tu foto</CardTitle>
        <CardDescription>
          Selecciona hasta {maxSelected} colores para crear una combinación.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-5 gap-2">
          {colors.map((c, index) => {
            const isSelected = selected.includes(c.value);
            return (
              <button
                key={c.value}
                type="button"
                onClick={() => onToggle(c.value)}
                aria-pressed={isSelected}
                aria-label={`Color ${index + 1} ${c.value} ${isSelected ? "seleccionado" : "disponible"}`}
                title={`${c.value}${isSelected ? " (seleccionado)" : ""}`}
                className={cn(
                  "group relative flex min-w-0 flex-col overflow-hidden rounded-lg border text-left transition-all",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                  isSelected ? "border-primary ring-2 ring-primary/30" : "border-border hover:border-primary/50"
                )}
              >
                <div
                  className="relative h-12 w-full sm:h-16"
                  style={{ backgroundColor: c.value }}
                >
                  {isSelected && (
                    <span className="absolute right-1 top-1 flex size-5 items-center justify-center rounded-full bg-primary text-primary-foreground">
                      <Check className="size-3.5" aria-hidden />
                    </span>
                  )}
                </div>
                <span className="block truncate px-1 py-1 text-center font-mono text-[11px] uppercase tracking-tight text-muted-foreground">
                  {c.value}
                </span>
              </button>
            );
          })}
        </div>
        <p className="mt-3 text-xs text-muted-foreground">
          {selected.length === 0
            ? `Aún no has elegido ningún color (máx. ${maxSelected})`
            : `${selected.length}/${maxSelected} ${selected.length === 1 ? "color elegido" : "colores elegidos"}`}
        </p>
      </CardContent>
    </Card>
  );
}
