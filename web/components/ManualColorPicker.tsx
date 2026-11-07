"use client";
import { useState } from "react";
import { toast } from "sonner";
import { Plus, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

type Props = {
  colors: string[];
  maxColors?: number;
  onAdd: (hex: string) => void;
  onRemove: (index: number) => void;
  onUpdate: (index: number, hex: string) => void;
};

function normalize(hex: string): string {
  return hex.trim().toLowerCase();
}

function isValidHex(hex: string): boolean {
  return /^#[0-9a-f]{6}$/i.test(hex.trim());
}

export function ManualColorPicker({ colors, maxColors = 4, onAdd, onRemove, onUpdate }: Props) {
  const [pickerValue, setPickerValue] = useState("#7c3aed");

  const handleAdd = () => {
    const hex = normalize(pickerValue);
    if (!isValidHex(hex)) {
      toast.error("Elige un color válido");
      return;
    }
    if (colors.map(normalize).includes(hex)) {
      toast.error("Ese color ya está en la lista");
      return;
    }
    if (colors.length >= maxColors) {
      toast.error(`Puedes seleccionar hasta ${maxColors} colores`);
      return;
    }
    onAdd(hex);
  };

  const handleEdit = (index: number, next: string) => {
    const hex = normalize(next);
    if (!isValidHex(hex)) return;
    if (colors.map(normalize).some((c, i) => i !== index && c === hex)) {
      toast.error("Ese color ya está en la lista");
      return;
    }
    onUpdate(index, hex);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Colores manuales</CardTitle>
        <CardDescription>
          Elige hasta {maxColors} colores con el selector y agrégalos a la lista.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-3">
          <input
            type="color"
            value={pickerValue}
            onChange={(e) => setPickerValue(e.target.value)}
            aria-label="Selector de color"
            className="h-10 w-14 cursor-pointer rounded-md border border-input bg-background p-1"
          />
          <span className="font-mono text-sm uppercase text-muted-foreground">
            {pickerValue}
          </span>
          <Button
            type="button"
            onClick={handleAdd}
            disabled={colors.length >= maxColors}
            className="ml-auto gap-1"
          >
            <Plus className="size-4" aria-hidden />
            Agregar
          </Button>
        </div>

        {colors.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Aún no has agregado ningún color (máx. {maxColors}).
          </p>
        ) : (
          <ul className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {colors.map((c, i) => (
              <li
                key={`${c}-${i}`}
                className="relative flex flex-col overflow-hidden rounded-lg border border-border"
              >
                <label
                  className="relative block h-14 w-full cursor-pointer sm:h-16"
                  style={{ backgroundColor: c }}
                  title={`Editar color ${c}`}
                >
                  <span className="sr-only">Editar color {i + 1} {c}</span>
                  <input
                    type="color"
                    value={c}
                    onChange={(e) => handleEdit(i, e.target.value)}
                    aria-label={`Editar color ${i + 1} ${c}`}
                    className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
                  />
                </label>
                <span className="block truncate px-1 py-1 text-center font-mono text-[11px] uppercase text-muted-foreground">
                  {c}
                </span>
                <button
                  type="button"
                  onClick={() => onRemove(i)}
                  aria-label={`Eliminar color ${c}`}
                  className="absolute right-1 top-1 flex size-6 items-center justify-center rounded-full bg-background/90 text-foreground shadow-sm transition-colors hover:bg-destructive hover:text-destructive-foreground"
                >
                  <X className="size-3.5" aria-hidden />
                </button>
              </li>
            ))}
          </ul>
        )}

        <p className="text-xs text-muted-foreground">
          {colors.length === 0
            ? `Aún no has elegido ningún color (máx. ${maxColors})`
            : `${colors.length}/${maxColors} ${colors.length === 1 ? "color elegido" : "colores elegidos"}. Pulsa un color para editarlo.`}
        </p>
      </CardContent>
    </Card>
  );
}
