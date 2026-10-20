"use client";
import Link from "next/link";
import { SwatchBook } from "lucide-react";
import { NavLinks } from "./NavLinks";

export function TopNav() {
  return (
    <header
      className="sticky top-0 z-40 hidden border-b border-transparent bg-background/60 backdrop-blur-xl supports-[backdrop-filter]:bg-background/60 md:flex"
      aria-label="Barra superior"
    >
      <div className="mx-auto flex h-14 w-full max-w-6xl items-center justify-between gap-4 px-4">
        <Link
          href="/"
          className="inline-flex items-center gap-2 font-heading text-base font-semibold tracking-tight"
        >
          <span className="inline-flex size-7 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <SwatchBook className="size-4" aria-hidden />
          </span>
          <span>Combina2</span>
        </Link>
        <NavLinks />
      </div>
    </header>
  );
}
