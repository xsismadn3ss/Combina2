import Link from "next/link";
import { Palette, ScanSearch } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "cn";

export default function HomePage() {
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="font-heading text-3xl font-semibold tracking-tight">Combina2</h1>
        <p className="text-sm text-muted-foreground">
          Descubre combinaciones de colores y crea paletas inspiradas en tus fotos favoritas.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card className="flex flex-col">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Palette className="size-5" /> Generar paleta
            </CardTitle>
            <CardDescription>
              Sube una foto, elige los colores que te gustan y crea propuestas para inspirarte.
            </CardDescription>
          </CardHeader>
          <CardContent className="mt-auto">
            <Link href="/generar" className={cn(buttonVariants(), "w-full")}>
              Ir a Generar
            </Link>
          </CardContent>
        </Card>

        <Card className="flex flex-col">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ScanSearch className="size-5" /> Identificar paleta
            </CardTitle>
            <CardDescription>
              Sube una foto para descubrir los colores y la armonía que la representan.
            </CardDescription>
          </CardHeader>
          <CardContent className="mt-auto">
            <Link href="/identificar" className={cn(buttonVariants({ variant: "secondary" }), "w-full")}>
              Ir a Identificar
            </Link>
          </CardContent>
        </Card>
      </div>

      <p className="text-xs text-muted-foreground">
        Ajusta tus preferencias de propuestas en{" "}
        <Link href="/config" className="underline underline-offset-4">
          Preferencias
        </Link>
        .
      </p>
    </div>
  );
}
