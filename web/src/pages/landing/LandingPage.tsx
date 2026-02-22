import { Link } from "react-router-dom";
import { ArrowRight, Cpu, ShieldAlert, Trophy, Zap } from "lucide-react";
import { motion } from "framer-motion";
import { useQuery } from "@tanstack/react-query";
import { AppShell } from "@/widgets/layout/AppShell";
import { fetchActiveSeason, fetchLeaderboard } from "@/features/ranking/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Button } from "@/shared/ui/button";
import { Badge } from "@/shared/ui/badge";

const FEATURE_CARDS = [
  {
    icon: Cpu,
    title: "IA real en combate",
    description: "Juega contra modelo o heuristica y siente la diferencia de estilo en tiempo real.",
  },
  {
    icon: Trophy,
    title: "Ranking competitivo",
    description: "Sigue temporadas activas, progreso ELO e historial de partidas por jugador.",
  },
  {
    icon: ShieldAlert,
    title: "Lore con identidad",
    description: "Una senal alienigena infecto la red. Tu mision: contener la expansion malware.",
  },
];

const QUICK_STATS = [
  { value: "2", label: "modos IA" },
  { value: "3", label: "skins activas" },
  { value: "100%", label: "enfoque tablero" },
];

const HOME_TOP_LIMIT = 3;

export function LandingPage(): JSX.Element {
  const seasonQuery = useQuery({
    queryKey: ["home-active-season"],
    queryFn: fetchActiveSeason,
  });

  const leaderboardQuery = useQuery({
    queryKey: ["home-top-leaderboard", seasonQuery.data?.id],
    queryFn: () => fetchLeaderboard(seasonQuery.data!.id, HOME_TOP_LIMIT, 0),
    enabled: Boolean(seasonQuery.data?.id),
  });

  const topPlayers = leaderboardQuery.data?.items ?? [];

  return (
    <AppShell>
      <div className="grid gap-4 lg:grid-cols-[1.25fr_0.75fr]">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, ease: "easeOut" }}
        >
          <Card className="overflow-hidden border-line/90 bg-[radial-gradient(circle_at_12%_6%,rgba(255,255,255,0.08),rgba(0,0,0,0)_35%),linear-gradient(155deg,rgba(255,255,255,0.02),rgba(0,0,0,0.35))]">
            <CardHeader className="relative">
              <motion.div
                className="pointer-events-none absolute right-4 top-4 h-12 w-12 rounded-full border border-primary/30"
                animate={{ scale: [0.95, 1.06, 0.95], opacity: [0.32, 0.72, 0.32] }}
                transition={{ duration: 2.8, repeat: Infinity, ease: "easeInOut" }}
              />
              <Badge variant="default" className="w-fit border-primary/35 bg-primary/12 text-primary">
                UNDERBYTELABS // ATAXX-ZERO
              </Badge>
              <CardTitle className="mt-3 text-3xl leading-tight sm:text-4xl">
                Arena tactica retro con IA moderna, ranking en vivo y estilo terminal elegante.
              </CardTitle>
              <CardDescription className="max-w-2xl">
                Elige tu modo, enfrenta a la entidad alienigena y mejora tu lectura de posicion turno a turno.
                Minimalista por fuera, intensa en el tablero.
              </CardDescription>
            </CardHeader>

            <CardContent className="space-y-4">
              <div className="flex flex-wrap gap-2">
                <Button asChild variant="default" className="group">
                  <Link to="/match">
                    Iniciar partida
                    <ArrowRight className="ml-2 h-4 w-4 transition-transform duration-200 group-hover:translate-x-1" />
                  </Link>
                </Button>
                <Button asChild variant="secondary">
                  <Link to="/ranking">Ver ranking</Link>
                </Button>
              </div>

              <div className="grid gap-2 rounded-lg border border-line/70 bg-black/35 p-3 sm:grid-cols-3">
                <p className="text-sm text-textDim">
                  <span className="font-semibold text-textMain">Modo rapido:</span> respuesta agil para partidas fluidas.
                </p>
                <p className="text-sm text-textDim">
                  <span className="font-semibold text-textMain">Modo fuerte:</span> mas calculo para jugadas exigentes.
                </p>
                <p className="text-sm text-textDim">
                  <span className="font-semibold text-textMain">Persistencia:</span> replay y seguimiento de estado.
                </p>
              </div>

              <div className="grid grid-cols-3 gap-2">
                {QUICK_STATS.map((item, idx) => (
                  <motion.div
                    key={item.label}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.22, delay: 0.08 + idx * 0.05 }}
                    className="rounded-lg border border-line/70 bg-black/45 px-3 py-2"
                  >
                    <p className="text-lg font-semibold text-textMain">{item.value}</p>
                    <p className="text-[11px] uppercase tracking-[0.12em] text-textDim">{item.label}</p>
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <div className="grid gap-4">
          {FEATURE_CARDS.map((item, idx) => (
            <motion.div
              key={item.title}
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.24, delay: 0.06 + idx * 0.05, ease: "easeOut" }}
              whileHover={{ y: -2 }}
            >
              <Card className="border-line/80 bg-black/72 transition-colors hover:border-primary/35">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <item.icon className="h-4 w-4 text-primary" />
                    {item.title}
                  </CardTitle>
                  <CardDescription>{item.description}</CardDescription>
                </CardHeader>
              </Card>
            </motion.div>
          ))}

          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.28, delay: 0.2, ease: "easeOut" }}
          >
            <Card className="border-primary/30 bg-primary/10">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base text-primary">
                  <Zap className="h-4 w-4" />
                  Consejo rapido
                </CardTitle>
                <CardDescription className="text-textMain/85">
                  Si eres nuevo, parte con heuristica normal. Si ya dominas, sube a modelo strong y compara replays.
                </CardDescription>
              </CardHeader>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.24, ease: "easeOut" }}
          >
            <Card className="border-line/80 bg-black/75">
              <CardHeader>
                <div className="flex items-center justify-between gap-2">
                  <CardTitle className="text-base">Temporada en vivo</CardTitle>
                  {seasonQuery.data?.name ? <Badge variant="success">{seasonQuery.data.name}</Badge> : null}
                </div>
                <CardDescription>Top jugadores detectados en la temporada activa.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                {seasonQuery.isLoading ? <p className="text-sm text-textDim">Cargando temporada...</p> : null}
                {seasonQuery.isError ? (
                  <p className="text-sm text-redGlow">No hay temporada activa disponible aun.</p>
                ) : null}
                {!seasonQuery.isLoading && !seasonQuery.isError && topPlayers.length === 0 ? (
                  <p className="text-sm text-textDim">Sin datos de ranking todavia.</p>
                ) : null}
                {topPlayers.map((entry) => (
                  <div
                    key={entry.user_id}
                    className="flex items-center justify-between rounded-md border border-line/70 bg-black/45 px-3 py-2"
                  >
                    <p className="text-sm text-textMain">
                      #{entry.rank} {entry.user_id.slice(0, 8)}
                    </p>
                    <p className="text-sm font-semibold text-primary">{entry.rating.toFixed(1)} ELO</p>
                  </div>
                ))}
                {leaderboardQuery.isLoading ? <p className="text-xs text-textDim">Actualizando top...</p> : null}
                {leaderboardQuery.isError ? (
                  <p className="text-xs text-redGlow">No se pudo cargar el top de jugadores.</p>
                ) : null}
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </div>
    </AppShell>
  );
}
