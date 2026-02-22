import { Navigate, Route, Routes } from "react-router-dom";
import { LandingPage } from "@/pages/landing/LandingPage";
import { RankingPage } from "@/pages/ranking/RankingPage";
import { MatchPage } from "@/pages/match/MatchPage";
import { LoginPage } from "@/pages/auth/LoginPage";
import { RegisterPage } from "@/pages/auth/RegisterPage";
import { ProfilePage } from "@/pages/profile/ProfilePage";
import { GameDetailPage } from "@/pages/profile/GameDetailPage";
import { RequireAuth, RequireGuest } from "@/app/router/guards";

export function AppRoutes(): JSX.Element {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/ranking" element={<RankingPage />} />
      <Route path="/match" element={<MatchPage />} />
      <Route
        path="/auth/login"
        element={
          <RequireGuest>
            <LoginPage />
          </RequireGuest>
        }
      />
      <Route
        path="/auth/register"
        element={
          <RequireGuest>
            <RegisterPage />
          </RequireGuest>
        }
      />
      <Route
        path="/profile"
        element={
          <RequireAuth>
            <ProfilePage />
          </RequireAuth>
        }
      />
      <Route
        path="/profile/games/:gameId"
        element={
          <RequireAuth>
            <GameDetailPage />
          </RequireAuth>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
