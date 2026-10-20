"use client";
import { useEffect, useState } from "react";

const STORAGE_KEY = "combina2:topK";
const DEFAULT = 5;

export function useTopK() {
  const [topK, setTopKState] = useState<number>(DEFAULT);

  useEffect(() => {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const n = parseInt(raw, 10);
      if (!Number.isNaN(n) && n >= 1 && n <= 10) setTopKState(n);
    }
  }, []);

  const setTopK = (value: number) => {
    const clamped = Math.min(10, Math.max(1, Math.round(value)));
    setTopKState(clamped);
    localStorage.setItem(STORAGE_KEY, String(clamped));
  };

  return { topK, setTopK };
}
