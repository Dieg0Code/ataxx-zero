import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg0: "var(--bg-0)",
        bg1: "var(--bg-1)",
        surface: "var(--surface)",
        line: "var(--line)",
        textMain: "var(--text-main)",
        textDim: "var(--text-dim)",
        primary: "var(--primary-green)",
        primarySoft: "var(--primary-green-soft)",
        amber: "var(--accent-amber)",
        orange: "var(--accent-orange)",
        redGlow: "var(--accent-red)",
        cyan: "var(--accent-cyan)"
      },
      boxShadow: {
        glow: "0 0 24px color-mix(in oklab, var(--primary-green) 35%, transparent)",
      },
    },
  },
  plugins: [],
};

export default config;
