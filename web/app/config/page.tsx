"use client";
import { useSyncExternalStore } from "react";
import { Settings } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { useTopK } from "@/hooks/useTopK";

const emptySubscribe = () => () => {};

export default function ConfigPage() {
  const { topK, setTopK } = useTopK();
  // true solo tras la hidratación: evita el mismatch con localStorage sin
  // setState dentro de un effect.
  const mounted = useSyncExternalStore(
    emptySubscribe,
    () => true,
    () => false
  );

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <div className="space-y-2">
        <h1 className="flex items-center gap-2 font-heading text-2xl font-semibold">
          <Settings className="size-6" /> Preferencias
        </h1>
        <p className="text-sm text-muted-foreground">Personaliza las propuestas que recibes.</p>
      </div>
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Número de propuestas</CardTitle>
          <CardDescription>Elige cuántas opciones de paleta quieres ver cada vez.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
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
