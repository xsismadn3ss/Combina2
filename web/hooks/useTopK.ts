"use client";

import { useSyncExternalStore } from "react";

const STORAGE_KEY = "combina2:topK";
const DEFAULT = 5;
const MIN = 1;
const MAX = 10;

function clampTopK(value: number): number {
  return Math.min(MAX, Math.max(MIN, Math.round(value)));
}

function readStoredTopK(): number {
  if (typeof window === "undefined") return DEFAULT;
  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (!raw) return DEFAULT;
  const n = parseInt(raw, 10);
  return Number.isNaN(n) ? DEFAULT : clampTopK(n);
}

// Store externo: el snapshot del servidor es siempre el default, así la
// hidratación no depende de localStorage. En el cliente React compara el
// snapshot del servidor con el de localStorage y actualiza tras hidratar.
let snapshot: number | null = null;
const listeners = new Set<() => void>();

function getSnapshot(): number {
  if (snapshot === null) snapshot = readStoredTopK();
  return snapshot;
}

function getServerSnapshot(): number {
  return DEFAULT;
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function setTopK(value: number): void {
  const clamped = clampTopK(value);
  snapshot = clamped;
  if (typeof window !== "undefined") {
    window.localStorage.setItem(STORAGE_KEY, String(clamped));
  }
  for (const listener of listeners) listener();
}

export function useTopK() {
  const topK = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  return { topK, setTopK };
}
