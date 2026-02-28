import { AppShell } from "@/widgets/layout/AppShell";
import { AuthCard } from "@/pages/auth/AuthCard";

export function RegisterPage(): JSX.Element {
  return (
    <AppShell>
      <div className="mx-auto max-w-md">
        <AuthCard initialTab="register" from="/profile" />
      </div>
    </AppShell>
  );
}

