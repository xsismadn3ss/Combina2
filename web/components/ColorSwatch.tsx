"use client";

import { useEffect, useRef, useState } from "react";
import { Check, Copy } from "lucide-react";
import { toast } from "sonner";
import { cn } from "cn";

import { copyTextToClipboard } from "@/lib/clipboard";

type Props = {
  value: string;
  /** Habilita el botón de copiar (solo en respuestas del modelo). */
  copyable?: boolean;
  className?: string;
  swatchClassName?: string;
};

/**
 * Cuadrado de color con su código HEX visible. En respuestas del modelo,
 * muestra un icono de copiar al hacer hover/focus y copia al hacer clic.
 */
export function ColorSwatch({
  value,
  copyable = false,
  className,
  swatchClassName = "h-12",
}: Props) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hex = value.toUpperCase();

  useEffect(() => {
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);

  const handleCopy = async () => {
    const ok = await copyTextToClipboard(hex);
    if (!ok) {
      toast.error("No pudimos copiar el color. Inténtalo de nuevo.");
      return;
    }
    setCopied(true);
    toast.success(`Color ${hex} copiado`);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setCopied(false), 1500);
  };

  const content = (
    <>
      <span
        className={cn("relative block w-full", swatchClassName)}
        style={{ backgroundColor: hex }}
      >
        {copyable ? (
          <span
            className={cn(
              "pointer-events-none absolute right-1 top-1 flex size-6 items-center justify-center rounded-md bg-background/85 text-foreground shadow-sm transition-opacity",
              copied
                ? "opacity-100"
                : "opacity-100 sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-visible:opacity-100"
            )}
            aria-hidden
          >
            {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
          </span>
        ) : null}
      </span>
      <span className="block truncate px-1 py-1 text-center font-mono text-[11px] uppercase tracking-tight text-muted-foreground">
        {hex}
      </span>
    </>
  );

  if (!copyable) {
    return (
      <div className={cn("flex min-w-0 flex-col overflow-hidden rounded-lg border", className)}>
        {content}
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={handleCopy}
      aria-label={`Copiar color ${hex}`}
      title={`Copiar ${hex}`}
      className={cn(
        "group flex min-w-0 flex-col overflow-hidden rounded-lg border text-left transition",
        "hover:border-primary/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1",
        className
      )}
    >
      {content}
    </button>
  );
}
