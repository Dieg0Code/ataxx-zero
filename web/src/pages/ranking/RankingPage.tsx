import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Crown,
  Loader2,
  Minus,
  Sparkles,
  Target,
  TrendingDown,
  TrendingUp,
  Trophy,
  UserRound,
} from "lucide-react";
import { motion } from "framer-motion";
import { AppShell } from "@/widgets/layout/AppShell";
import { useAuth } from "@/app/providers/useAuth";
import { fetchActiveSeason, fetchLeaderboard, fetchUserRating } from "@/features/ranking/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Button } from "@/shared/ui/button";
import { Badge } from "@/shared/ui/badge";

const PAGE_SIZE = 10;

export function RankingPage(): JSX.Element {
  const [offset, setOffset] = useState(0);
  const [competitorFilter, setCompetitorFilter] = useState<"all" | "humans" | "bots">("all");
  const [search, setSearch] = useState("");
  const { user, isAuthenticated } = useAuth();

  const seasonQuery = useQuery({
    queryKey: ["activeSeason"],
    queryFn: fetchActiveSeason,
  });

  const seasonId = seasonQuery.data?.id;
  const leaderboardQuery = useQuery({
    queryKey: ["leaderboard", seasonId, offset, competitorFilter, search],
    queryFn: () =>
      fetchLeaderboard(seasonId!, PAGE_SIZE, offset, {
        competitorFilter,
        query: search,
      }),
    enabled: Boolean(seasonId),
  });

  const myRatingQuery = useQuery({
    queryKey: ["my-rating", user?.id, seasonId],
    queryFn: () => fetchUserRating(user!.id, seasonId!),
    enabled: Boolean(user?.id && seasonId),
  });

  const entries = useMemo(() => leaderboardQuery.data?.items ?? [], [leaderboardQuery.data]);
  const myVisibleEntry = useMemo(
    () => entries.find((entry) => entry.user_id === user?.id),
    [entries, user?.id],
  );
  const isInitialLeaderboardLoading = leaderboardQuery.isLoading && entries.length === 0;
  const showEmptyState = !leaderboardQuery.isLoading && !leaderboardQuery.isError && entries.length === 0;
  const emptyStateMessage = search.trim().length > 0
    ? `No hay resultados para "${search.trim()}".`
    : competitorFilter === "bots"
      ? "No hay bots en esta temporada."
      : competitorFilter === "humans"
        ? "No hay jugadores humanos en esta temporada."
        : "Aun no hay jugadores en el ranking.";

  return (
    <AppShell>
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25 }}
      >
        <div className={`grid gap-4 ${isAuthenticated ? "xl:grid-cols-[320px_1fr]" : "xl:grid-cols-1"}`}>
          {isAuthenticated ? (
            <Card className="border-zinc-800/90 bg-zinc-950/60">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <Target className="h-4.5 w-4.5 text-lime-300" />
                  Tu posicion
                </CardTitle>
                <CardDescription>Lectura rapida de tu estado competitivo actual.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {myRatingQuery.isLoading ? (
                  <div className="space-y-2">
                    <div className="h-14 animate-pulse rounded-md border border-line bg-bg1/70" />
                    <div className="grid grid-cols-2 gap-2">
                      <div className="h-12 animate-pulse rounded-md border border-line bg-bg1/70" />
                      <div className="h-12 animate-pulse rounded-md border border-line bg-bg1/70" />
                    </div>
                  </div>
                ) : null}

                {myRatingQuery.data ? (
                  <>
                    <div className="rounded-md border border-lime-400/40 bg-lime-400/10 p-3 shadow-[0_0_24px_rgba(163,230,53,0.12)]">
                      <p className="text-[11px] uppercase tracking-[0.14em] text-textDim">Liga actual</p>
                      <div className="mt-1 flex items-center justify-between">
                        <p className="flex items-center gap-2 text-base font-semibold text-lime-200">
                          <Sparkles className="h-4 w-4 text-lime-300" />
                          {myRatingQuery.data.league} {myRatingQuery.data.division}
                        </p>
                        <Badge
                          variant="success"
                          className="border-lime-300/70 bg-lime-300/25 text-lime-100 shadow-[0_0_18px_rgba(163,230,53,0.28)]"
                        >
                          {myRatingQuery.data.lp} LP
                        </Badge>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      <div className="rounded-md border border-line bg-bg1/70 px-3 py-2">
                        <p className="text-[11px] uppercase tracking-[0.12em] text-textDim">MMR</p>
                        <p className="mt-1 text-sm font-semibold text-textMain">
                          {myRatingQuery.data.rating.toFixed(1)}
                        </p>
                      </div>
                      <div className="rounded-md border border-line bg-bg1/70 px-3 py-2">
                        <p className="text-[11px] uppercase tracking-[0.12em] text-textDim">V/D/E</p>
                        <p className="mt-1 text-sm font-semibold text-textMain">
                          {myRatingQuery.data.wins}/{myRatingQuery.data.losses}/{myRatingQuery.data.draws}
                        </p>
                      </div>
                    </div>

                    <p className="text-xs text-textDim">
                      {myVisibleEntry
                        ? `Tu ranking visible: #${myVisibleEntry.rank}`
                        : "No apareces en esta pagina del top todavia."}
                    </p>
                  </>
                ) : null}

                {myRatingQuery.isError ? (
                  <p className="text-sm text-redGlow">No se pudo cargar tu estado competitivo.</p>
                ) : null}
              </CardContent>
            </Card>
          ) : null}

          <Card className="border-zinc-800/90 bg-zinc-950/60">
            <CardHeader>
              <div className="flex items-center justify-between gap-2">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <Trophy className="h-5 w-5 text-amber" />
                    Ranking publico
                  </CardTitle>
                  <CardDescription>Clasificacion en vivo de la temporada activa.</CardDescription>
                </div>
                {seasonQuery.data ? <Badge variant="success">{seasonQuery.data.name}</Badge> : null}
              </div>
            </CardHeader>
            <CardContent>
              {seasonQuery.isLoading ? <p className="text-sm text-textDim">Cargando temporada...</p> : null}
              {seasonQuery.isError ? <p className="text-sm text-redGlow">Aun no hay temporada activa.</p> : null}

              <div className="mt-3 overflow-hidden rounded-lg border border-line">
                {isInitialLeaderboardLoading ? (
                  <div className="h-0.5 w-full overflow-hidden bg-line/30">
                    <motion.div
                      className="h-full w-1/3 bg-primary/80"
                      animate={{ x: ["-120%", "320%"] }}
                      transition={{ duration: 1.1, repeat: Infinity, ease: "linear" }}
                    />
                  </div>
                ) : null}
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line bg-bg1/70 px-3 py-2">
                  <div className="inline-flex rounded-md border border-line bg-zinc-950/70 p-1">
                    <button
                      type="button"
                      className={`rounded px-2 py-1 text-xs ${
                        competitorFilter === "all" ? "bg-primary/20 text-primary" : "text-textDim hover:text-textMain"
                      }`}
                      onClick={() => {
                        setOffset(0);
                        setCompetitorFilter("all");
                      }}
                    >
                      Global
                    </button>
                    <button
                      type="button"
                      className={`rounded px-2 py-1 text-xs ${
                        competitorFilter === "humans"
                          ? "bg-primary/20 text-primary"
                          : "text-textDim hover:text-textMain"
                      }`}
                      onClick={() => {
                        setOffset(0);
                        setCompetitorFilter("humans");
                      }}
                    >
                      Humanos
                    </button>
                    <button
                      type="button"
                      className={`rounded px-2 py-1 text-xs ${
                        competitorFilter === "bots" ? "bg-primary/20 text-primary" : "text-textDim hover:text-textMain"
                      }`}
                      onClick={() => {
                        setOffset(0);
                        setCompetitorFilter("bots");
                      }}
                    >
                      Bots
                    </button>
                  </div>
                  <input
                    type="search"
                    value={search}
                    onChange={(event) => {
                      setOffset(0);
                      setSearch(event.target.value);
                    }}
                    placeholder="Buscar jugador..."
                    className="h-8 w-full max-w-[260px] rounded-md border border-line bg-zinc-950/80 px-2 text-xs text-textMain outline-none transition focus:border-primary/60"
                  />
                </div>
                <table className="w-full text-left text-sm">
                  <thead className="sticky top-0 z-10 bg-bg1 text-textDim">
                    <tr>
                      <th className="px-3 py-2">#</th>
                      <th className="px-3 py-2">Jugador</th>
                      <th className="px-3 py-2">Trend</th>
                      <th className="px-3 py-2">Liga</th>
                      <th className="px-3 py-2">LP</th>
                      <th className="px-3 py-2">ELO</th>
                      <th className="px-3 py-2">V/D/E</th>
                    </tr>
                  </thead>
                  <tbody>
                    {entries.map((entry) => (
                      <tr
                        key={entry.user_id}
                        className={`border-t border-line/70 ${
                          user?.id === entry.user_id ? "bg-lime-400/10" : "bg-transparent"
                        }`}
                      >
                        <td className="px-3 py-2 text-amber">{entry.rank}</td>
                        <td className="px-3 py-2 text-textMain">
                          <span className="inline-flex items-center gap-2">
                            {entry.rank === 1 ? <Crown className="h-3.5 w-3.5 text-lime-300" /> : null}
                            {entry.username ?? entry.user_id.slice(0, 8)}
                            {entry.rank === 1 ? (
                              <Badge variant="success" className="border-lime-300/40 bg-lime-300/20 text-[10px]">
                                {entry.prestige_title ?? "Singularity"}
                              </Badge>
                            ) : null}
                            {entry.is_bot ? (
                              <Badge variant="default" className="text-[10px]">
                                Bot{entry.bot_kind ? `:${entry.bot_kind}` : ""}
                              </Badge>
                            ) : null}
                            {user?.id === entry.user_id ? (
                              <Badge variant="default" className="text-[10px]">
                                <UserRound className="mr-1 h-3 w-3" />
                                Tu
                              </Badge>
                            ) : null}
                          </span>
                        </td>
                        <td className="px-3 py-2">
                          {entry.recent_lp_delta === null || entry.recent_lp_delta === undefined ? (
                            <span className="text-textDim">-</span>
                          ) : entry.recent_lp_delta > 0 ? (
                            <span className="inline-flex items-center gap-1 text-lime-300">
                              <TrendingUp className="h-3.5 w-3.5" />+{entry.recent_lp_delta} LP
                            </span>
                          ) : entry.recent_lp_delta < 0 ? (
                            <span className="inline-flex items-center gap-1 text-red-300">
                              <TrendingDown className="h-3.5 w-3.5" />
                              {entry.recent_lp_delta} LP
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 text-textDim">
                              <Minus className="h-3.5 w-3.5" />0 LP
                            </span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-textMain">
                          {entry.league} {entry.division}
                        </td>
                        <td className="px-3 py-2 text-primary">{entry.lp}</td>
                        <td className="px-3 py-2 text-primary">{entry.rating.toFixed(1)}</td>
                        <td className="px-3 py-2 text-textDim">
                          {entry.wins}/{entry.losses}/{entry.draws}
                        </td>
                      </tr>
                    ))}
                    {isInitialLeaderboardLoading ? (
                      <tr className="border-t border-line/70">
                        <td colSpan={7} className="px-3 py-7">
                          <div className="inline-flex items-center gap-2 text-sm text-textDim">
                            <Loader2 className="h-4 w-4 animate-spin text-primary" />
                            Sincronizando ranking...
                          </div>
                        </td>
                      </tr>
                    ) : null}
                    {showEmptyState ? (
                      <tr className="border-t border-line/70">
                        <td colSpan={7} className="px-3 py-7">
                          <p className="text-sm text-textDim">{emptyStateMessage}</p>
                        </td>
                      </tr>
                    ) : null}
                  </tbody>
                </table>
                {leaderboardQuery.isError ? (
                  <p className="px-3 py-3 text-sm text-redGlow">No se pudo cargar el ranking.</p>
                ) : null}
              </div>

              <div className="mt-4 flex items-center justify-between">
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  disabled={offset === 0}
                  onClick={() => setOffset((prev) => Math.max(0, prev - PAGE_SIZE))}
                >
                  Anterior
                </Button>
                <p className="text-xs text-textDim">desplazamiento: {offset}</p>
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  disabled={!leaderboardQuery.data?.has_more}
                  onClick={() => setOffset((prev) => prev + PAGE_SIZE)}
                >
                  Siguiente
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </motion.div>
    </AppShell>
  );
}
