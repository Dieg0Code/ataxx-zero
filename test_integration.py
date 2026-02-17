from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    _ensure_src_on_path()
    from data.dataset import AtaxxDataset
    from data.replay_buffer import ReplayBuffer
    from engine.mcts import MCTS
    from game.actions import ACTION_SPACE
    from game.board import AtaxxBoard
    from game.constants import PLAYER_1, PLAYER_2
    from model.system import AtaxxZero
    from model.transformer import AtaxxTransformerNet

    print("== Integration Test: Ataxx Zero ==")

    board = AtaxxBoard()
    _check(board.grid[0, 0] == PLAYER_1, "Initial P1 piece missing at (0,0)")
    _check(board.grid[6, 6] == PLAYER_1, "Initial P1 piece missing at (6,6)")
    _check(board.grid[0, 6] == PLAYER_2, "Initial P2 piece missing at (0,6)")
    _check(board.grid[6, 0] == PLAYER_2, "Initial P2 piece missing at (6,0)")

    obs = board.get_observation()
    _check(obs.shape == (3, 7, 7), f"Bad observation shape: {obs.shape}")
    _check(obs.dtype == np.float32, f"Bad observation dtype: {obs.dtype}")

    model = AtaxxTransformerNet(d_model=64, nhead=8, num_layers=2, dim_feedforward=128)
    x = torch.from_numpy(obs).unsqueeze(0).repeat(4, 1, 1, 1)
    pi_logits, value = model(x)
    _check(
        pi_logits.shape == (4, ACTION_SPACE.num_actions),
        f"Bad policy shape: {pi_logits.shape}",
    )
    _check(value.shape == (4, 1), f"Bad value shape: {value.shape}")

    mcts = MCTS(model=model, c_puct=1.5, n_simulations=32, device="cpu")
    probs = mcts.run(board=AtaxxBoard(), add_dirichlet_noise=False, temperature=1.0)
    _check(
        probs.shape == (ACTION_SPACE.num_actions,),
        f"Bad probs shape: {probs.shape}",
    )
    _check(float(np.sum(probs)) > 0.99, "MCTS probs are not normalized")

    buffer = ReplayBuffer(capacity=100)
    for _ in range(16):
        obs_dummy = np.random.randn(3, 7, 7).astype(np.float32)
        pi_dummy = np.random.rand(ACTION_SPACE.num_actions).astype(np.float32)
        pi_dummy /= float(np.sum(pi_dummy))
        value_dummy = float(np.random.choice([-1.0, 0.0, 1.0]))
        buffer.save_game([(obs_dummy, pi_dummy, value_dummy)])

    dataset = AtaxxDataset(buffer=buffer, augment=True, reference_buffer=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    boards, target_pis, target_vs = next(iter(loader))
    _check(boards.shape == (8, 3, 7, 7), f"Bad boards batch: {boards.shape}")
    _check(
        target_pis.shape == (8, ACTION_SPACE.num_actions),
        f"Bad policy batch: {target_pis.shape}",
    )
    _check(target_vs.shape == (8,), f"Bad value batch: {target_vs.shape}")

    system = AtaxxZero(
        learning_rate=1e-3,
        d_model=64,
        nhead=8,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    )
    loss = system.training_step((boards, target_pis, target_vs), batch_idx=0)
    _check(isinstance(loss, torch.Tensor), "Loss is not a tensor")
    _check(loss.ndim == 0, f"Loss is not scalar: ndim={loss.ndim}")
    _check(float(loss.item()) > 0.0, f"Loss must be > 0, got {loss.item()}")

    print("All integration checks passed.")


if __name__ == "__main__":
    main()
