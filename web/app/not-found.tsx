import type { Metadata } from "next";
import Link from "next/link";
import { Ghost } from "lucide-react";

import { buttonVariants } from "@/components/ui/button";

export const metadata: Metadata = {
  title: "Página no encontrada | Combina2",
};

export default function NotFound() {
  return (
    <div className="flex min-h-[calc(100svh-9rem)] flex-col items-center justify-center gap-6 px-4 text-center">
      <Ghost aria-hidden strokeWidth={1.5} className="rainbow-icon size-24 sm:size-32" />
      <h1 className="font-heading text-3xl font-semibold sm:text-4xl">Página no encontrada</h1>
      <Link href="/" className={buttonVariants({ size: "lg" })}>
        Volver a inicio
      </Link>
    </div>
  );
}
