import { useLocation } from "react-router-dom";
import { AppShell } from "@/widgets/layout/AppShell";
import { AuthCard } from "@/pages/auth/AuthCard";

type LocationState = {
  from?: string;
};

export function LoginPage(): JSX.Element {
  const location = useLocation();
  const from = (location.state as LocationState | null)?.from ?? "/profile";

  return (
    <AppShell>
      <div className="mx-auto max-w-md">
        <AuthCard initialTab="login" from={from} />
      </div>
    </AppShell>
  );
}

