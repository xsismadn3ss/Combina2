"use client";
import { useCallback, useRef, useState } from "react";
import { toast } from "sonner";
import { Pencil, Plus, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "cn";

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

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

/** #rrggbb -> {h: 0-360, s: 0-1, v: 0-1}. Null si inválido. */
function hexToHsv(hex: string): { h: number; s: number; v: number } | null {
  const m = /^#([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return null;
  const n = parseInt(m[1], 16);
  const r = ((n >> 16) & 255) / 255;
  const g = ((n >> 8) & 255) / 255;
  const b = (n & 255) / 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const d = max - min;
  let h = 0;
  if (d !== 0) {
    if (max === r) h = ((g - b) / d) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  const s = max === 0 ? 0 : d / max;
  return { h, s, v: max };
}

function hsvToHex(h: number, s: number, v: number): string {
  const c = v * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = v - c;
  let r = 0;
  let g = 0;
  let b = 0;
  if (h < 60) { r = c; g = x; }
  else if (h < 120) { r = x; g = c; }
  else if (h < 180) { g = c; b = x; }
  else if (h < 240) { g = x; b = c; }
  else if (h < 300) { r = x; b = c; }
  else { r = c; b = x; }
  const to = (v2: number) => Math.round((v2 + m) * 255).toString(16).padStart(2, "0");
  return `#${to(r)}${to(g)}${to(b)}`;
}

const PRESETS = [
  "#ef4444", "#f97316", "#f59e0b", "#84cc16", "#22c55e", "#14b8a6",
  "#06b6d4", "#3b82f6", "#6366f1", "#8b5cf6", "#d946ef", "#ec4899",
  "#ffffff", "#9ca3af", "#4b5563", "#000000",
];

export function ManualColorPicker({ colors, maxColors = 4, onAdd, onRemove, onUpdate }: Props) {
  const [hsv, setHsv] = useState({ h: 262, s: 0.8, v: 0.93 });
  const [hexText, setHexText] = useState("#7c3aed");
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const planeRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);

  const currentHex = hsvToHex(hsv.h, hsv.s, hsv.v);

  const syncFromHexText = (text: string) => {
    setHexText(text);
    const parsed = hexToHsv(text);
    if (parsed) setHsv(parsed);
  };

  const pickFromPlane = useCallback(
    (clientX: number, clientY: number) => {
      const el = planeRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const s = clamp01((clientX - rect.left) / rect.width);
      const v = clamp01(1 - (clientY - rect.top) / rect.height);
      setHsv((prev) => {
        const next = { ...prev, s, v };
        setHexText(hsvToHex(next.h, next.s, next.v));
        return next;
      });
    },
    []
  );

  const handleAddOrSave = () => {
    const hex = normalize(hexText);
    if (!isValidHex(hex)) {
      toast.error("Escribe un color válido, por ejemplo #7c3aed");
      return;
    }
    if (editingIndex !== null) {
      if (colors.map(normalize).some((c, i) => i !== editingIndex && c === hex)) {
        toast.error("Ese color ya está en la lista");
        return;
      }
      onUpdate(editingIndex, hex);
      setEditingIndex(null);
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

  const startEdit = (index: number) => {
    const parsed = hexToHsv(colors[index]);
    if (parsed) setHsv(parsed);
    setHexText(colors[index].toLowerCase());
    setEditingIndex(index);
  };

  const cancelEdit = () => setEditingIndex(null);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Colores manuales</CardTitle>
        <CardDescription>
          Elige hasta {maxColors} colores con el selector de esta página y agrégalos a la lista.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Plano saturación/valor: todo en página, sin diálogos del sistema */}
        <div
          ref={planeRef}
          role="slider"
          tabIndex={0}
          aria-label="Selector de saturación y brillo"
          aria-valuetext={currentHex}
          onPointerDown={(e) => {
            draggingRef.current = true;
            (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
            pickFromPlane(e.clientX, e.clientY);
          }}
          onPointerMove={(e) => {
            if (draggingRef.current) pickFromPlane(e.clientX, e.clientY);
          }}
          onPointerUp={() => { draggingRef.current = false; }}
          onPointerCancel={() => { draggingRef.current = false; }}
          onKeyDown={(e) => {
            const step = e.shiftKey ? 0.1 : 0.02;
            if (e.key === "ArrowLeft") { setHsv((p) => ({ ...p, s: clamp01(p.s - step) })); e.preventDefault(); }
            else if (e.key === "ArrowRight") { setHsv((p) => ({ ...p, s: clamp01(p.s + step) })); e.preventDefault(); }
            else if (e.key === "ArrowUp") { setHsv((p) => ({ ...p, v: clamp01(p.v + step) })); e.preventDefault(); }
            else if (e.key === "ArrowDown") { setHsv((p) => ({ ...p, v: clamp01(p.v - step) })); e.preventDefault(); }
          }}
          className="relative h-44 w-full cursor-crosshair touch-none rounded-lg border border-input focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          style={{
            background: `linear-gradient(to top, rgb(0,0,0), transparent), linear-gradient(to right, rgb(255,255,255), transparent), hsl(${Math.round(hsv.h)}, 100%, 50%)`,
          }}
        >
          <span
            aria-hidden
            className="pointer-events-none absolute size-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white shadow"
            style={{ left: `${hsv.s * 100}%`, top: `${(1 - hsv.v) * 100}%`, backgroundColor: currentHex }}
          />
        </div>

        {/* Tono */}
        <div className="space-y-1">
          <label htmlFor="manual-hue" className="text-xs font-medium text-muted-foreground">
            Tono
          </label>
          <input
            id="manual-hue"
            type="range"
            min={0}
            max={360}
            step={1}
            value={Math.round(hsv.h)}
            onChange={(e) => {
              const h = Number(e.target.value);
              setHsv((prev) => {
                const next = { ...prev, h };
                setHexText(hsvToHex(next.h, next.s, next.v));
                return next;
              });
            }}
            aria-label="Tono del color"
            className="w-full accent-primary"
            style={{
              background: "linear-gradient(to right, #f00, #ff0, #0f0, #0ff, #00f, #f0f, #f00)",
              borderRadius: 6,
              height: 10,
            }}
          />
        </div>

        {/* Vista previa + HEX + acción */}
        <div className="flex items-center gap-3">
          <span
            aria-hidden
            className="h-10 w-14 shrink-0 rounded-md border border-input"
            style={{ backgroundColor: currentHex }}
          />
          <Input
            value={hexText}
            onChange={(e) => syncFromHexText(e.target.value)}
            aria-label="Código hexadecimal del color"
            placeholder="#7c3aed"
            maxLength={7}
            spellCheck={false}
            autoComplete="off"
            className="font-mono uppercase"
          />
          <Button
            type="button"
            onClick={handleAddOrSave}
            disabled={editingIndex === null && colors.length >= maxColors}
            className="ml-auto shrink-0 gap-1"
          >
            {editingIndex !== null ? <Pencil className="size-4" aria-hidden /> : <Plus className="size-4" aria-hidden />}
            {editingIndex !== null ? "Guardar" : "Agregar"}
          </Button>
        </div>
        {editingIndex !== null && (
          <Button type="button" variant="ghost" size="sm" onClick={cancelEdit} className="h-auto p-0 text-xs">
            Cancelar edición
          </Button>
        )}

        {/* Presets rápidos: botones en página */}
        <div className="space-y-1">
          <p className="text-xs font-medium text-muted-foreground">Colores rápidos</p>
          <div className="flex flex-wrap gap-1.5">
            {PRESETS.map((p) => (
              <button
                key={p}
                type="button"
                onClick={() => {
                  const parsed = hexToHsv(p);
                  if (parsed) setHsv(parsed);
                  setHexText(p);
                }}
                aria-label={`Usar color ${p}`}
                title={p}
                className="size-7 rounded-md border border-input transition-transform hover:scale-110 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                style={{ backgroundColor: p }}
              />
            ))}
          </div>
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
                className={cn(
                  "relative flex flex-col overflow-hidden rounded-lg border",
                  editingIndex === i ? "border-primary ring-2 ring-primary/30" : "border-border"
                )}
              >
                <button
                  type="button"
                  onClick={() => startEdit(i)}
                  aria-label={`Editar color ${i + 1} ${c}`}
                  title="Pulsar para editar en el selector"
                  className="block h-14 w-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring sm:h-16"
                  style={{ backgroundColor: c }}
                />
                <span className="block truncate px-1 py-1 text-center font-mono text-[11px] uppercase text-muted-foreground">
                  {c}
                </span>
                <button
                  type="button"
                  onClick={() => {
                    if (editingIndex === i) setEditingIndex(null);
                    onRemove(i);
                  }}
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
            : `${colors.length}/${maxColors} ${colors.length === 1 ? "color elegido" : "colores elegidos"}. Pulsa un color para editarlo en el selector.`}
        </p>
      </CardContent>
    </Card>
  );
}
