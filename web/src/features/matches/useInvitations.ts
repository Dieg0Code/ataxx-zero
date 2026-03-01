import { useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  acceptInvitation,
  fetchIncomingInvitations,
  openInvitationsSocket,
  rejectInvitation,
  type HumanInvitation,
  type InvitationsWsEvent,
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
  scope,
  fallbackPollingMs = 0,
}: UseInvitationsOptions): UseInvitationsResult {
  const queryClient = useQueryClient();
  const [liveInvitations, setLiveInvitations] = useState<HumanInvitation[] | null>(null);
  const [actionLoadingId, setActionLoadingId] = useState<string | null>(null);

  const invitationsQuery = useQuery({
    queryKey: ["invitations", scope, accessToken],
    queryFn: () => fetchIncomingInvitations(accessToken!, 12, 0),
    enabled: enabled && includeInitialFetch && accessToken !== null,
    refetchInterval:
      enabled && includeInitialFetch && fallbackPollingMs > 0
        ? fallbackPollingMs
        : false,
    refetchIntervalInBackground: true,
  });

  useEffect(() => {
    if (!enabled || accessToken === null) {
      setLiveInvitations(null);
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

    const scheduleReconnect = (): void => {
      if (cancelled) {
        return;
      }
      clearReconnectTimer();
      reconnectTimer = window.setTimeout(() => {
        if (!cancelled) {
          connectSocket();
        }
      }, 1200);
    };

    const onEvent = (event: InvitationsWsEvent): void => {
      if (event.type !== "invitations.status") {
        return;
      }
      const pendingItems = event.payload.items.filter((item) => item.status === "pending");
      setLiveInvitations(pendingItems);
    };

    const onSocketFailure = (): void => {
      setLiveInvitations(null);
      if (includeInitialFetch) {
        void queryClient.invalidateQueries({ queryKey: ["invitations", scope, accessToken] });
      }
      scheduleReconnect();
    };

    const connectSocket = (): void => {
      if (cancelled) {
        return;
      }
      socket = openInvitationsSocket(accessToken, onEvent);
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
  }, [accessToken, enabled, includeInitialFetch, queryClient, scope]);

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
        await queryClient.invalidateQueries({ queryKey: ["invitations", scope, accessToken] });
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
        await queryClient.invalidateQueries({ queryKey: ["invitations", scope, accessToken] });
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
