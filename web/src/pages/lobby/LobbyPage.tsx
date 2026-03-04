import { useCallback } from "react";
import { Link } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowRight, Bot, Eye, Settings2, Users } from "lucide-react";
import { useAuth } from "@/app/providers/useAuth";
import { prefetchPublicPlayers } from "@/features/identity/api";
import { AppShell } from "@/widgets/layout/AppShell";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";

function LobbyOptionCard({
  badge,
  title,
  description,
  to,
  cta,
  icon,
  onPrefetch,
}: {
  badge: string;
  title: string;
  description: string;
  to: string;
  cta: string;
  icon: JSX.Element;
  onPrefetch: () => void;
}): JSX.Element {
  return (
    <motion.div
      className="rounded-xl border border-zinc-800/80 bg-zinc-950/65 p-3"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, ease: "easeOut" }}
    >
      <p className="inline-flex items-center gap-1 text-[11px] uppercase tracking-[0.14em] text-zinc-500">
        {icon}
        {badge}
      </p>
      <h3 className="mt-1 text-base font-semibold text-zinc-100">{title}</h3>
      <p className="mt-1 text-sm text-zinc-400">{description}</p>
      <Button asChild size="sm" className="mt-3 w-full border border-lime-300/70 bg-lime-300 text-black hover:bg-lime-200">
        <Link to={to} onMouseEnter={onPrefetch} onFocus={onPrefetch} onTouchStart={onPrefetch}>
          {cta}
          <ArrowRight className="ml-2 h-4 w-4" />
        </Link>
      </Button>
    </motion.div>
  );
}

export function LobbyPage(): JSX.Element {
  const { isAuthenticated, accessToken } = useAuth();
  const prefetchSetup = useCallback((): void => {
    if (!isAuthenticated || accessToken === null) {
      return;
    }
    void prefetchPublicPlayers(accessToken, { limit: 200 });
  }, [accessToken, isAuthenticated]);

  return (
    <AppShell>
      <div className="grid gap-4 lg:grid-cols-[1.15fr_0.85fr]">
        <Card className="border-zinc-800/90 bg-[linear-gradient(180deg,rgba(132,204,22,0.04),rgba(0,0,0,0.38))] shadow-[0_0_16px_rgba(132,204,22,0.08)]">
          <CardHeader>
            <Badge variant="default" className="w-fit border-lime-300/45 bg-lime-300/10 text-lime-200">
              SALA PERSONALIZADA
            </Badge>
            <CardTitle className="mt-2 text-2xl">Crear partida</CardTitle>
            <CardDescription>Invita jugadores, crea duelos contra bots o lanza simulaciones IA vs IA.</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3 md:grid-cols-3">
            <LobbyOptionCard
              badge="1v1"
              title="Sala Humano vs Humano"
              description="Invita un rival humano y espera su aceptacion en la sala."
              to="/match?setup=invite"
              cta="Abrir sala 1v1"
              icon={<Users className="h-3.5 w-3.5 text-lime-300" />}
              onPrefetch={prefetchSetup}
            />
            <LobbyOptionCard
              badge="BOT"
              title="Duelo vs Bot"
              description="Configura una partida casual para practicar sin cola competitiva."
              to="/match?setup=bot"
              cta="Crear duelo vs bot"
              icon={<Bot className="h-3.5 w-3.5 text-lime-300" />}
              onPrefetch={prefetchSetup}
            />
            <LobbyOptionCard
              badge="SPECTATE"
              title="Simulacion IA vs IA"
              description="Observa dos agentes en combate para analizar estilo y decisiones."
              to="/match?setup=spectate"
              cta="Lanzar simulacion"
              icon={<Eye className="h-3.5 w-3.5 text-lime-300" />}
              onPrefetch={prefetchSetup}
            />
          </CardContent>
        </Card>

        <AnimatePresence>
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2, delay: 0.04 }}>
            <Card className="border-zinc-800/90 bg-zinc-950/60">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <Settings2 className="h-4 w-4 text-zinc-400" />
                  Ajustes avanzados
                </CardTitle>
                <CardDescription>
                  La configuracion detallada queda en perfil para mantener esta pantalla enfocada en crear sala rapido.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button asChild variant="ghost" size="sm" className="h-8 border border-zinc-800/70 bg-transparent px-2 text-zinc-400 hover:bg-zinc-900/70 hover:text-zinc-100">
                  <Link to="/profile#laboratorio">Ir a configuracion avanzada</Link>
                </Button>
              </CardContent>
            </Card>
          </motion.div>
        </AnimatePresence>
      </div>
    </AppShell>
  );
}

