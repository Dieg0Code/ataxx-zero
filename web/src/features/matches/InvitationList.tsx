import { Loader2, X } from "lucide-react";
import { Button } from "@/shared/ui/button";
import { type HumanInvitation } from "@/features/matches/api";

type InvitationListVariant = "panel" | "card";

type InvitationListProps = {
  items: HumanInvitation[];
  isLoading: boolean;
  isError: boolean;
  actionLoadingId: string | null;
  onAccept: (gameId: string) => void;
  onReject: (gameId: string) => void;
  variant?: InvitationListVariant;
  emptyText?: string;
  loadingText?: string;
  errorText?: string;
};

function itemClassName(variant: InvitationListVariant): string {
  if (variant === "panel") {
    return "rounded-md border border-zinc-800 bg-zinc-950/70 px-2 py-2";
  }
  return "rounded-md border border-zinc-700/80 bg-zinc-950/75 px-3 py-2";
}

export function InvitationList({
  items,
  isLoading,
  isError,
  actionLoadingId,
  onAccept,
  onReject,
  variant = "card",
  emptyText = "No tienes invitaciones pendientes.",
  loadingText = "Cargando invitaciones...",
  errorText = "No se pudieron cargar las invitaciones.",
}: InvitationListProps): JSX.Element {
  return (
    <>
      {isLoading ? <p className="text-sm text-textDim">{loadingText}</p> : null}
      {isError ? <p className="text-sm text-redGlow">{errorText}</p> : null}
      {!isLoading && !isError && items.length === 0 ? (
        <p className="text-sm text-textDim">{emptyText}</p>
      ) : null}
      {items
        .filter((invitation) => invitation.status === "pending")
        .map((invitation) => {
          const isLoadingAction = actionLoadingId === invitation.id;
          return (
            <div key={invitation.id} className={itemClassName(variant)}>
              <p className="text-[11px] uppercase tracking-[0.12em] text-textDim">Duelo pendiente</p>
              <p className="mt-1 font-mono text-xs text-textMain">{invitation.id.slice(0, 8)}</p>
              <div className="mt-2 grid grid-cols-2 gap-2">
                <Button
                  type="button"
                  size="sm"
                  variant="secondary"
                  className="border border-lime-300/55 bg-lime-300/14 text-lime-100 hover:bg-lime-300/22"
                  disabled={actionLoadingId !== null}
                  onClick={() => onAccept(invitation.id)}
                >
                  {isLoadingAction ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "Aceptar"}
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="secondary"
                  className="border border-red-500/55 bg-red-500/12 text-red-200 hover:bg-red-500/22"
                  disabled={actionLoadingId !== null}
                  onClick={() => onReject(invitation.id)}
                >
                  {isLoadingAction ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : variant === "panel" ? (
                    <X className="h-3.5 w-3.5" />
                  ) : (
                    "Rechazar"
                  )}
                </Button>
              </div>
            </div>
          );
        })}
    </>
  );
}
