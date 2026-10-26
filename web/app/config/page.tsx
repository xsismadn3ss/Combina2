"use client";
import { useSyncExternalStore } from "react";
import { Monitor, Moon, Settings, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ButtonGroup } from "@/components/ui/button-group";
import { useTopK } from "@/hooks/useTopK";

const emptySubscribe = () => () => {};

const THEME_OPTIONS = [
  { value: "light", label: "Claro", icon: Sun },
  { value: "dark", label: "Oscuro", icon: Moon },
  { value: "system", label: "Sistema", icon: Monitor },
] as const;

export default function ConfigPage() {
  const { topK, setTopK } = useTopK();
  const { theme, setTheme } = useTheme();
  // true solo tras la hidratación: evita el mismatch con localStorage sin
  // setState dentro de un effect.
  const mounted = useSyncExternalStore(
    emptySubscribe,
    () => true,
    () => false
  );

  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-6">
      <div className="flex flex-col gap-2">
        <h1 className="flex items-center gap-2 font-heading text-2xl font-semibold">
          <Settings className="size-6" /> Preferencias
        </h1>
        <p className="text-sm text-muted-foreground">Personaliza el tema y las propuestas que recibes.</p>
      </div>
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Tema</CardTitle>
          <CardDescription>Elige cómo se ve la interfaz.</CardDescription>
        </CardHeader>
        <CardContent>
          <ButtonGroup aria-label="Tema de la interfaz" className="w-full">
            {THEME_OPTIONS.map(({ value, label, icon: Icon }) => {
              const active = mounted && theme === value;
              return (
                <Button
                  key={value}
                  variant={active ? "default" : "outline"}
                  size="sm"
                  className="flex-1"
                  aria-pressed={active}
                  onClick={() => setTheme(value)}
                >
                  <Icon data-icon="inline-start" aria-hidden />
                  {label}
                </Button>
              );
            })}
          </ButtonGroup>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Número de propuestas</CardTitle>
          <CardDescription>Elige cuántas opciones de paleta quieres ver cada vez.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <Label htmlFor="proposal-count">Cantidad</Label>
            <Badge variant="outline">Número de propuestas: {mounted ? topK : "…"}</Badge>
          </div>
          <Slider
            id="proposal-count"
            value={[mounted ? topK : 5]}
            min={1}
            max={10}
            step={1}
            onValueChange={(v) => {
              const arr = Array.isArray(v) ? v : [v as number];
              setTopK(arr[0] ?? 5);
            }}
            aria-label="Número de propuestas"
          />
          <div className="flex justify-between text-xs text-muted-foreground"><span>1 propuesta</span><span>10 propuestas</span></div>
        </CardContent>
      </Card>
    </div>
  );
}
