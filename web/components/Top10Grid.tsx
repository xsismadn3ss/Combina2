"use client";
import { Check } from "lucide-react";
import type { Color } from "@/lib/api/types";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { cn } from "cn";

type Props = {
  colors: Color[];
  selected: string[];
  onToggle: (hex: string) => void;
};

export function Top10Grid({ colors, selected, onToggle }: Props) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Colores de tu foto</CardTitle>
        <CardDescription>
          Selecciona uno o más colores para crear una combinación.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
          {colors.map((c, index) => {
            const isSelected = selected.includes(c.value);
            return (
              <button
                key={c.value}
                type="button"
                onClick={() => onToggle(c.value)}
                aria-pressed={isSelected}
                aria-label={`Color ${index + 1} ${isSelected ? "seleccionado" : "disponible"}`}
                className={cn(
                  "group relative flex flex-col overflow-hidden rounded-xl border text-left transition-all",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                  isSelected ? "border-primary ring-2 ring-primary/30" : "border-border hover:border-primary/50"
                )}
              >
                <div
                  className="h-20 w-full"
                  style={{ backgroundColor: c.value }}
                />
                <div className="flex flex-1 flex-col gap-1 p-2">
                  <span className="text-xs font-medium">Color {index + 1}</span>
                </div>
                {isSelected && (
                  <span className="absolute right-1.5 top-1.5 flex size-6 items-center justify-center rounded-full bg-primary text-primary-foreground">
                    <Check className="size-4" aria-hidden />
                  </span>
                )}
              </button>
            );
          })}
        </div>
        <p className="mt-3 text-xs text-muted-foreground">
          {selected.length === 0 ? "Aún no has elegido ningún color" : `${selected.length} ${selected.length === 1 ? "color elegido" : "colores elegidos"}`}
        </p>
      </CardContent>
    </Card>
  );
}
