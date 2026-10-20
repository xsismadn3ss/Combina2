"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { House, Palette, ScanSearch, Settings } from "lucide-react";
import { cn } from "cn";

export type NavItem = {
  href: string;
  label: string;
  icon: React.ElementType;
};

export const NAV_ITEMS: NavItem[] = [
  { href: "/", label: "Home", icon: House },
  { href: "/generar", label: "Generar", icon: Palette },
  { href: "/identificar", label: "Identificar", icon: ScanSearch },
  { href: "/config", label: "Config", icon: Settings },
];

export function NavLinks({
  onNavigate,
  orientation = "horizontal",
}: {
  onNavigate?: () => void;
  orientation?: "horizontal" | "vertical";
}) {
  const pathname = usePathname();

  return (
    <nav
      aria-label="Navegación principal"
      className={cn(
        "flex gap-1",
        orientation === "vertical" ? "flex-col" : "flex-row items-center"
      )}
    >
      {NAV_ITEMS.map((item) => {
        const isActive =
          item.href === "/" ? pathname === "/" : pathname?.startsWith(item.href);
        const Icon = item.icon;
        return (
          <Link
            key={item.href}
            href={item.href}
            onClick={onNavigate}
            aria-current={isActive ? "page" : undefined}
            className={cn(
              "inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
              "hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
              isActive
                ? "bg-primary text-primary-foreground hover:bg-primary/90"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <Icon className="size-4 shrink-0" aria-hidden />
            <span>{item.label}</span>
          </Link>
        );
      })}
    </nav>
  );
}
