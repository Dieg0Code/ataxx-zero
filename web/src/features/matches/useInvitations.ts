import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  acceptInvitation,
  fetchIncomingInvitations,
  openInvitationsSocket,
  rejectInvitation,
  type HumanInvitation,
  type InvitationsWsEvent,
  type PagedInvitations,
} from "@/features/matches/api";

type UseInvitationsOptions = {
  accessToken: string | null;
  enabled: boolean;
  includeInitialFetch: boolean;
  scope: string;
  fallbackPollingMs?: number;
};

type UseInvitationsResult = {
  invitations: HumanInvitation[];
  isLoading: boolean;
  isError: boolean;
  actionLoadingId: string | null;
  acceptInvitationById: (gameId: string) => Promise<void>;
  rejectInvitationById: (gameId: string) => Promise<void>;
};

export function useInvitations({
  accessToken,
  enabled,
  includeInitialFetch,
  scope: _scope,
  fallbackPollingMs = 0,
}: UseInvitationsOptions): UseInvitationsResult {
  void _scope;
  const queryClient = useQueryClient();
  const [liveInvitations, setLiveInvitations] = useState<HumanInvitation[] | null>(null);
  const [actionLoadingId, setActionLoadingId] = useState<string | null>(null);
  const [socketHealthy, setSocketHealthy] = useState(false);
  const reconnectAttemptRef = useRef(0);
  const lastInvalidateAtRef = useRef(0);
  const queryKey = useMemo(() => ["invitations", accessToken] as const, [accessToken]);

  const invitationsQuery = useQuery({
    queryKey,
    queryFn: () => fetchIncomingInvitations(accessToken!, 12, 0),
    enabled: enabled && includeInitialFetch && accessToken !== null,
    // Poll only while websocket fallback is degraded; this keeps UI fresh
    // without hammering the API every few seconds during healthy socket sessions.
    refetchInterval:
      enabled && includeInitialFetch && fallbackPollingMs > 0 && !socketHealthy
        ? fallbackPollingMs
        : false,
    refetchIntervalInBackground: true,
    staleTime: 15_000,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  });

  useEffect(() => {
    if (!enabled || accessToken === null) {
      setLiveInvitations(null);
      setSocketHealthy(false);
      return;
    }
    let cancelled = false;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | null = null;

    const clearReconnectTimer = (): void => {
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    const maybeInvalidateInvitations = (): void => {
      if (!includeInitialFetch) {
        return;
      }
      const nowMs = Date.now();
      // If socket flaps, one refresh is enough; repeated invalidations only add API noise.
      if (nowMs - lastInvalidateAtRef.current < 10_000) {
        return;
      }
      lastInvalidateAtRef.current = nowMs;
      void queryClient.invalidateQueries({ queryKey });
    };

    const scheduleReconnect = (): void => {
      if (cancelled) {
        return;
      }
      clearReconnectTimer();
      // Backoff avoids reconnect storms when the websocket backend is transiently down.
      const attempt = reconnectAttemptRef.current;
      const delayMs = Math.min(12_000, 1_200 * 2 ** attempt);
      reconnectAttemptRef.current = Math.min(attempt + 1, 4);
      reconnectTimer = window.setTimeout(() => {
        if (!cancelled) {
          connectSocket();
        }
      }, delayMs);
    };

    const onEvent = (event: InvitationsWsEvent): void => {
      if (event.type !== "invitations.status") {
        return;
      }
      setSocketHealthy(true);
      reconnectAttemptRef.current = 0;
      const pendingItems = event.payload.items.filter((item) => item.status === "pending");
      setLiveInvitations(pendingItems);
      const total =
        typeof event.payload.total === "number" && Number.isFinite(event.payload.total)
          ? event.payload.total
          : pendingItems.length;
      if (includeInitialFetch) {
        queryClient.setQueryData<PagedInvitations>(queryKey, (previous) => {
          const limit = previous?.limit ?? 12;
          const offset = previous?.offset ?? 0;
          return {
            items: pendingItems,
            total,
            limit,
            offset,
            has_more: total > offset + limit,
          };
        });
      }
    };

    const onSocketFailure = (): void => {
      setSocketHealthy(false);
      setLiveInvitations(null);
      maybeInvalidateInvitations();
      scheduleReconnect();
    };

    const connectSocket = (): void => {
      if (cancelled) {
        return;
      }
      socket = openInvitationsSocket(accessToken, onEvent);
      socket.onopen = () => {
        setSocketHealthy(true);
        reconnectAttemptRef.current = 0;
      };
      socket.onerror = () => {
        onSocketFailure();
      };
      socket.onclose = () => {
        onSocketFailure();
      };
    };

    connectSocket();

    return () => {
      cancelled = true;
      clearReconnectTimer();
      socket?.close();
    };
  }, [accessToken, enabled, includeInitialFetch, queryClient, queryKey]);

  const invitations = useMemo(
    () => (liveInvitations ?? invitationsQuery.data?.items ?? []).filter((item) => item.status === "pending"),
    [invitationsQuery.data?.items, liveInvitations],
  );

  const acceptInvitationById = async (gameId: string): Promise<void> => {
    if (accessToken === null) {
      throw new Error("Debes iniciar sesion para aceptar invitaciones.");
    }
    setActionLoadingId(gameId);
    try {
      await acceptInvitation(accessToken, gameId);
      setLiveInvitations((prev) => (prev === null ? prev : prev.filter((item) => item.id !== gameId)));
      if (includeInitialFetch) {
        await queryClient.invalidateQueries({ queryKey });
      }
    } finally {
      setActionLoadingId(null);
    }
  };

  const rejectInvitationById = async (gameId: string): Promise<void> => {
    if (accessToken === null) {
      throw new Error("Debes iniciar sesion para rechazar invitaciones.");
    }
    setActionLoadingId(gameId);
    try {
      await rejectInvitation(accessToken, gameId);
      setLiveInvitations((prev) => (prev === null ? prev : prev.filter((item) => item.id !== gameId)));
      if (includeInitialFetch) {
        await queryClient.invalidateQueries({ queryKey });
      }
    } finally {
      setActionLoadingId(null);
    }
  };

  return {
    invitations,
    isLoading: includeInitialFetch ? invitationsQuery.isLoading : false,
    isError: includeInitialFetch ? invitationsQuery.isError : false,
    actionLoadingId,
    acceptInvitationById,
    rejectInvitationById,
  };
}
