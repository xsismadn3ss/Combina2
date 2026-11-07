"use client";

import { useSyncExternalStore } from "react";

const STORAGE_KEY = "combina2-snow-enabled";
const DEFAULT = true;

function readStored(): boolean {
  if (typeof window === "undefined") return DEFAULT;
  return window.localStorage.getItem(STORAGE_KEY) !== "0";
}

// Store externo: el snapshot del servidor es siempre el default, así la
// hidratación no depende de localStorage. En el cliente React compara el
// snapshot del servidor con el de localStorage y actualiza tras hidratar.
let snapshot: boolean | null = null;
const listeners = new Set<() => void>();

function getSnapshot(): boolean {
  if (snapshot === null) snapshot = readStored();
  return snapshot;
}

function getServerSnapshot(): boolean {
  return DEFAULT;
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function setEnabled(value: boolean): void {
  snapshot = value;
  if (typeof window !== "undefined") {
    window.localStorage.setItem(STORAGE_KEY, value ? "1" : "0");
  }
  for (const listener of listeners) listener();
}

export function useSnow() {
  const enabled = useSyncExternalStore(
    subscribe,
    getSnapshot,
    getServerSnapshot
  );
  return { enabled, setEnabled };
}
