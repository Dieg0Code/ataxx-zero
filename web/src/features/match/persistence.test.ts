import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPersistedGame } from "@/features/match/persistence";

const apiGetMock = vi.fn();
const apiPostMock = vi.fn();

vi.mock("@/shared/api/client", () => ({
  apiGet: (...args: unknown[]) => apiGetMock(...args),
  apiPost: (...args: unknown[]) => apiPostMock(...args),
}));

describe("createPersistedGame", () => {
  beforeEach(() => {
    apiGetMock.mockReset();
    apiPostMock.mockReset();
  });

  it("creates ranked game with rated=true when user plays vs bot", async () => {
    apiGetMock
      .mockResolvedValueOnce({ id: "season-1" })
      .mockResolvedValueOnce({
        items: [
          {
            user_id: "bot-h-1",
            agent_type: "heuristic",
            heuristic_level: "normal",
            model_mode: null,
            enabled: true,
          },
        ],
      });
    apiPostMock.mockResolvedValue({ id: "game-1" });

    await createPersistedGame("token", "heuristic", {
      ranked: true,
      preferredHeuristicLevel: "normal",
      player1Agent: "human",
      player2Agent: "heuristic",
    });

    expect(apiPostMock).toHaveBeenCalledWith(
      "/api/v1/gameplay/games",
      expect.objectContaining({
        season_id: "season-1",
        queue_type: "ranked",
        rated: true,
        player1_agent: "human",
        player2_agent: "heuristic",
        player2_id: "bot-h-1",
      }),
      expect.any(Object),
    );
  });

  it("throws if ranked IA vs IA is requested", async () => {
    await expect(
      createPersistedGame("token", "heuristic", {
        ranked: true,
        player1Agent: "heuristic",
        player2Agent: "heuristic",
      }),
    ).rejects.toThrow("no puede jugar ranked");
  });

  it("does not call backend when ranked IA vs IA is blocked locally", async () => {
    await expect(
      createPersistedGame("token", "heuristic", {
        ranked: true,
        player1Agent: "heuristic",
        player2Agent: "heuristic",
      }),
    ).rejects.toThrow("no puede jugar ranked");
    expect(apiGetMock).not.toHaveBeenCalled();
    expect(apiPostMock).not.toHaveBeenCalled();
  });
});
