import { createContext, useContext, useEffect, useMemo, useState } from "react";

export type SkinName = "terminal-neo" | "amber-crt" | "oxide-red";

type ThemeContextValue = {
  skin: SkinName;
  setSkin: (skin: SkinName) => void;
};

const ThemeContext = createContext<ThemeContextValue | null>(null);

const THEME_STORAGE_KEY = "ataxx.skin";
const THEMES: SkinName[] = ["terminal-neo", "amber-crt", "oxide-red"];

function isTheme(value: string | null): value is SkinName {
  return value !== null && THEMES.includes(value as SkinName);
}

export function ThemeProvider({ children }: { children: React.ReactNode }): JSX.Element {
  const [skin, setSkinState] = useState<SkinName>("terminal-neo");

  useEffect(() => {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (isTheme(stored)) setSkinState(stored);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = skin;
    localStorage.setItem(THEME_STORAGE_KEY, skin);
  }, [skin]);

  const value = useMemo<ThemeContextValue>(
    () => ({
      skin,
      setSkin: setSkinState,
    }),
    [skin]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used inside ThemeProvider.");
  return ctx;
}
