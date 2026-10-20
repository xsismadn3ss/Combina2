"use client";
import { useState } from "react";
import { SwatchBook, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
  DrawerDescription,
  DrawerClose,
} from "@/components/ui/drawer";
import { NavLinks } from "./NavLinks";

export function MobileDock() {
  const [open, setOpen] = useState(false);

  return (
    <>
      {/* md:hidden -> solo mobile */}
      <div
        className="fixed bottom-4 left-1/2 z-40 flex -translate-x-1/2 items-center gap-3 rounded-full border bg-background/80 px-3 py-2 shadow-lg backdrop-blur-xl md:hidden"
        aria-label="Navegación móvil"
      >
        <span className="px-2 text-sm font-medium tracking-tight">Combina2</span>
        <Button
          aria-label="Abrir menú de navegación"
          aria-expanded={open}
          aria-controls="mobile-drawer"
          onClick={() => setOpen(true)}
          className="size-14 rounded-2xl bg-primary p-0 text-primary-foreground hover:bg-primary/90"
        >
          <SwatchBook className="size-6" aria-hidden />
        </Button>
      </div>

      <Drawer open={open} onOpenChange={setOpen}>
        <DrawerContent
          id="mobile-drawer"
          aria-label="Menú de navegación"
          className="md:hidden"
        >
          <DrawerHeader className="flex flex-row items-center justify-between">
            <div className="flex flex-col gap-1 text-left">
              <DrawerTitle>Navegación</DrawerTitle>
              <DrawerDescription>Elige una sección para continuar</DrawerDescription>
            </div>
            <DrawerClose className="inline-flex size-8 items-center justify-center rounded-md hover:bg-muted">
              <X className="size-4" />
              <span className="sr-only">Cerrar menú</span>
            </DrawerClose>
          </DrawerHeader>
          <div className="p-4 pb-8">
            <NavLinks orientation="vertical" onNavigate={() => setOpen(false)} />
          </div>
        </DrawerContent>
      </Drawer>
    </>
  );
}
