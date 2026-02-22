import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Trophy } from "lucide-react";
import { motion } from "framer-motion";
import { AppShell } from "@/widgets/layout/AppShell";
import { fetchActiveSeason, fetchLeaderboard } from "@/features/ranking/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Button } from "@/shared/ui/button";
import { Badge } from "@/shared/ui/badge";

const PAGE_SIZE = 10;

export function RankingPage(): JSX.Element {
  const [offset, setOffset] = useState(0);

  const seasonQuery = useQuery({
    queryKey: ["activeSeason"],
    queryFn: fetchActiveSeason,
  });

  const seasonId = seasonQuery.data?.id;
  const leaderboardQuery = useQuery({
    queryKey: ["leaderboard", seasonId, offset],
    queryFn: () => fetchLeaderboard(seasonId!, PAGE_SIZE, offset),
    enabled: Boolean(seasonId),
  });

  const entries = useMemo(() => leaderboardQuery.data?.items ?? [], [leaderboardQuery.data]);

  return (
    <AppShell>
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
        <Card>
        <CardHeader>
          <div className="flex items-center justify-between gap-2">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Trophy className="h-5 w-5 text-amber" />
                Ranking público
              </CardTitle>
              <CardDescription>Clasificación en vivo de la temporada activa.</CardDescription>
            </div>
            {seasonQuery.data && <Badge variant="success">{seasonQuery.data.name}</Badge>}
          </div>
        </CardHeader>
        <CardContent>
          {seasonQuery.isLoading && <p className="text-sm text-textDim">Cargando temporada...</p>}
          {seasonQuery.isError && <p className="text-sm text-redGlow">Aún no hay temporada activa.</p>}

          <div className="mt-3 overflow-hidden rounded-lg border border-line">
            <table className="w-full text-left text-sm">
              <thead className="bg-bg1 text-textDim">
                <tr>
                  <th className="px-3 py-2">#</th>
                  <th className="px-3 py-2">Jugador</th>
                  <th className="px-3 py-2">ELO</th>
                  <th className="px-3 py-2">V/D/E</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((entry) => (
                  <tr key={entry.user_id} className="border-t border-line/70">
                    <td className="px-3 py-2 text-amber">{entry.rank}</td>
                    <td className="px-3 py-2 text-textMain">{entry.user_id.slice(0, 8)}</td>
                    <td className="px-3 py-2 text-primary">{entry.rating.toFixed(1)}</td>
                    <td className="px-3 py-2 text-textDim">
                      {entry.wins}/{entry.losses}/{entry.draws}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {leaderboardQuery.isLoading && (
              <p className="px-3 py-3 text-sm text-textDim">Cargando tabla...</p>
            )}
            {leaderboardQuery.isError && (
              <p className="px-3 py-3 text-sm text-redGlow">No se pudo cargar el ranking.</p>
            )}
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
      </motion.div>
    </AppShell>
  );
}
