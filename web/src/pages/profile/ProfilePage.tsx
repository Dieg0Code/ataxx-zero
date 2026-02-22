import { useMemo, useState } from "react";
import { Link, Navigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { AppShell } from "@/widgets/layout/AppShell";
import { useAuth } from "@/app/providers/useAuth";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { fetchMyGames, type ProfileGame } from "@/features/profile/api";

const PAGE_SIZE = 8;

function outcomeLabel(game: ProfileGame, userId: string): string {
  if (game.status !== "finished") {
    return "En curso";
  }
  if (game.winner_user_id === null) {
    return "Empate";
  }
  return game.winner_user_id === userId ? "Victoria" : "Derrota";
}

function queueLabel(queueType: string): string {
  return queueType === "ranked" ? "Ranked" : "Casual";
}

export function ProfilePage(): JSX.Element {
  const { user, loading, logout, accessToken } = useAuth();
  const [offset, setOffset] = useState(0);

  const createdAt = useMemo(() => {
    if (!user?.created_at) {
      return "-";
    }
    return new Date(user.created_at).toLocaleString();
  }, [user?.created_at]);

  const gamesQuery = useQuery({
    queryKey: ["profile-games", offset, accessToken],
    queryFn: () => fetchMyGames(accessToken!, PAGE_SIZE, offset),
    enabled: Boolean(accessToken),
  });

  if (loading) {
    return (
      <AppShell>
        <p className="text-sm text-textDim">Cargando perfil...</p>
      </AppShell>
    );
  }
  if (user === null) {
    return <Navigate to="/auth/login" replace />;
  }

  return (
    <AppShell>
      <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
        <Card className="lg:row-span-2">
          <CardHeader>
            <CardTitle>Mi perfil</CardTitle>
            <CardDescription>Tu identidad de jugador y estado de cuenta.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <p>
              <span className="text-textDim">Usuario:</span> <span className="text-textMain">{user.username}</span>
            </p>
            <p>
              <span className="text-textDim">Email:</span> <span className="text-textMain">{user.email ?? "-"}</span>
            </p>
            <p>
              <span className="text-textDim">ID:</span> <span className="text-textMain">{user.id}</span>
            </p>
            <p>
              <span className="text-textDim">Creado:</span> <span className="text-textMain">{createdAt}</span>
            </p>
            <div className="flex gap-2 pt-1">
              <Badge variant={user.is_active ? "success" : "warning"}>
                {user.is_active ? "Activo" : "Inactivo"}
              </Badge>
              {user.is_admin ? <Badge variant="default">Admin</Badge> : null}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Acciones</CardTitle>
          </CardHeader>
          <CardContent>
            <Button
              type="button"
              variant="secondary"
              className="w-full"
              onClick={() => {
                void logout();
              }}
            >
              Cerrar sesion
            </Button>
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Historial de partidas</CardTitle>
            <CardDescription>Partidas visibles para tu cuenta autenticada.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {gamesQuery.isLoading ? <p className="text-sm text-textDim">Cargando partidas...</p> : null}
            {gamesQuery.isError ? (
              <p className="text-sm text-redGlow">No se pudo cargar el historial de partidas.</p>
            ) : null}
            {!gamesQuery.isLoading && !gamesQuery.isError && (gamesQuery.data?.items.length ?? 0) === 0 ? (
              <p className="text-sm text-textDim">Todavia no tienes partidas registradas.</p>
            ) : null}
            {(gamesQuery.data?.items ?? []).map((game) => (
              <div
                key={game.id}
                className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-line/70 bg-black/40 px-3 py-2 text-sm"
              >
                <div className="min-w-[180px]">
                  <p className="text-textMain">{game.id.slice(0, 8)}</p>
                  <p className="text-xs text-textDim">
                    {queueLabel(game.queue_type)} · {game.rated ? "con ELO" : "sin ELO"}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant={game.status === "finished" ? "success" : "warning"}>
                    {outcomeLabel(game, user.id)}
                  </Badge>
                  <Badge variant="default">{game.status}</Badge>
                  <Button asChild size="sm" variant="secondary">
                    <Link to={`/profile/games/${game.id}`}>Detalle</Link>
                  </Button>
                </div>
              </div>
            ))}

            <div className="flex items-center justify-between pt-1">
              <Button
                type="button"
                variant="secondary"
                size="sm"
                disabled={offset === 0}
                onClick={() => setOffset((prev) => Math.max(0, prev - PAGE_SIZE))}
              >
                Anterior
              </Button>
              <p className="text-xs text-textDim">offset: {offset}</p>
              <Button
                type="button"
                variant="secondary"
                size="sm"
                disabled={!gamesQuery.data?.has_more}
                onClick={() => setOffset((prev) => prev + PAGE_SIZE)}
              >
                Siguiente
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </AppShell>
  );
}
