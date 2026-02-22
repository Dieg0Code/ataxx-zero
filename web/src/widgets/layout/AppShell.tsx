import { Link, useLocation } from "react-router-dom";
import { Search } from "lucide-react";
import { motion } from "framer-motion";
import { useTheme, type SkinName } from "@/app/providers/ThemeProvider";
import { useAuth } from "@/app/providers/useAuth";
import { Button } from "@/shared/ui/button";
import { Input } from "@/shared/ui/input";
import { Badge } from "@/shared/ui/badge";
import { cn } from "@/shared/lib/utils";

const NAV_ITEMS = [
  { to: "/", label: "Inicio" },
  { to: "/match", label: "Jugar" },
  { to: "/ranking", label: "Ranking" },
  { to: "/profile", label: "Perfil" },
];

const SKINS: SkinName[] = ["terminal-neo", "amber-crt", "oxide-red"];

export function AppShell({ children }: { children: React.ReactNode }): JSX.Element {
  const { skin, setSkin } = useTheme();
  const { isAuthenticated, user, logout } = useAuth();
  const location = useLocation();

  return (
    <div className="mx-auto min-h-screen w-full max-w-[1320px] px-3 py-3 sm:px-5">
      <motion.header
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.28, ease: "easeOut" }}
        className="rounded-2xl border border-line/80 bg-black/50 px-4 py-3 backdrop-blur-xl"
      >
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="h-7 w-7 rounded-full bg-gradient-to-br from-primary via-cyan to-amber" />
            <div>
              <p className="text-sm font-semibold text-textMain">underbyteLabs</p>
              <p className="text-xs text-textDim">ataxx-zero arena</p>
            </div>
            <Badge variant="success">beta</Badge>
          </div>

          <div className="flex items-center gap-2">
            <div className="relative hidden sm:block">
              <Search className="pointer-events-none absolute left-2.5 top-2 h-4 w-4 text-textDim" />
              <Input className="w-56 pl-8" placeholder="Buscar..." />
            </div>
            <select
              className="h-9 rounded-md border border-line bg-bg1 px-3 text-sm text-textMain focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
              value={skin}
              onChange={(e) => setSkin(e.target.value as SkinName)}
            >
              {SKINS.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
            <Button size="sm" variant="secondary">
              Sugerencias
            </Button>
            {isAuthenticated ? (
              <Button
                size="sm"
                variant="secondary"
                onClick={() => {
                  void logout();
                }}
              >
                Salir ({user?.username ?? "user"})
              </Button>
            ) : (
              <Button size="sm" asChild variant="secondary">
                <Link to="/auth/login">Entrar</Link>
              </Button>
            )}
          </div>
        </div>

        <nav className="mt-3 flex items-center gap-1 overflow-x-auto border-t border-line/70 pt-2">
          {NAV_ITEMS.map((item) => {
            const active = location.pathname === item.to;
            return (
              <Button
                key={item.to}
                asChild
                size="sm"
                variant="ghost"
                className={cn(
                  "rounded-md px-3 transition-transform duration-200 hover:-translate-y-0.5",
                  active && "bg-white/10 text-textMain",
                  !active && "text-textDim",
                )}
              >
                <Link to={item.to}>{item.label}</Link>
              </Button>
            );
          })}
        </nav>
      </motion.header>
      <motion.section
        key={location.pathname}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.24, ease: "easeOut" }}
        className="mt-4"
      >
        {children}
      </motion.section>
    </div>
  );
}
