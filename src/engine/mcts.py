from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from game.actions import ACTION_SPACE
from game.board import AtaxxBoard


class MCTSNode:
    """Search tree node with AlphaZero statistics."""

    def __init__(self, prior: float) -> None:
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children: dict[int, MCTSNode] = {}

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """AlphaZero-style Monte Carlo Tree Search."""

    def __init__(
        self,
        model: nn.Module,
        c_puct: float = 1.5,
        n_simulations: int = 400,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.c_puct = c_puct
        self.n_simulations = n_simulations

    def run(
        self,
        board: AtaxxBoard,
        add_dirichlet_noise: bool = False,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Run MCTS from `board` and return visit-based policy probabilities.
        """
        probs = np.zeros(ACTION_SPACE.num_actions, dtype=np.float32)
        if board.is_game_over():
            return probs

        root = MCTSNode(prior=1.0)
        self._expand(root, board)

        if add_dirichlet_noise and root.children:
            self._add_dirichlet_noise(root, alpha=0.3, frac=0.25)

        for _ in range(self.n_simulations):
            node = root
            scratch_board = board.copy()
            search_path = [node]

            while node.children:
                action_idx, node = self._select_child(node)
                move = ACTION_SPACE.decode(action_idx)
                scratch_board.step(move)
                search_path.append(node)

            if scratch_board.is_game_over():
                value = self._terminal_value_for_current_player(scratch_board)
            else:
                value = self._expand(node, scratch_board)

            self._backpropagate(search_path, value)

        return self._get_action_probs(root, temperature)

    def _add_dirichlet_noise(self, node: MCTSNode, alpha: float, frac: float) -> None:
        actions = list(node.children.keys())
        noise = np.random.dirichlet([alpha] * len(actions))
        for idx, action_idx in enumerate(actions):
            child = node.children[action_idx]
            child.prior = (1.0 - frac) * child.prior + frac * float(noise[idx])

    def _select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        best_action = -1
        best_child: MCTSNode | None = None
        best_score = -float("inf")
        sqrt_parent = math.sqrt(node.visit_count + 1)

        for action_idx, child in node.children.items():
            # child.value() is from child-player perspective; negate for parent.
            q_value = -child.value()
            u_value = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_action = action_idx
                best_child = child

        if best_child is None:
            raise RuntimeError("No child selected from a non-empty node.")
        return best_action, best_child

    def _expand(self, node: MCTSNode, board: AtaxxBoard) -> float:
        """
        Expand a leaf node and return value in current-player perspective.
        """
        valid_moves = board.get_valid_moves()
        include_pass = len(valid_moves) == 0
        action_mask_np = ACTION_SPACE.mask_from_moves(valid_moves, include_pass=include_pass)

        obs = board.get_observation()
        state_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        action_mask = torch.from_numpy(action_mask_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value_tensor = self.model(state_tensor, action_mask=action_mask)

        policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        value = float(value_tensor.item())

        legal_action_indices = np.flatnonzero(action_mask_np > 0)
        if legal_action_indices.size == 0:
            return value

        priors = policy[legal_action_indices]
        prior_sum = float(np.sum(priors))

        if prior_sum <= 0.0:
            uniform_prior = 1.0 / float(legal_action_indices.size)
            for action_idx in legal_action_indices:
                node.children[int(action_idx)] = MCTSNode(prior=uniform_prior)
            return value

        for action_idx, prior in zip(legal_action_indices, priors, strict=True):
            node.children[int(action_idx)] = MCTSNode(prior=float(prior / prior_sum))
        return value

    def _terminal_value_for_current_player(self, board: AtaxxBoard) -> float:
        winner = board.get_result()
        if winner == 0:
            return 0.0
        return 1.0 if winner == board.current_player else -1.0

    def _backpropagate(self, path: list[MCTSNode], value: float) -> None:
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _get_action_probs(self, root: MCTSNode, temperature: float) -> np.ndarray:
        probs = np.zeros(ACTION_SPACE.num_actions, dtype=np.float32)
        if not root.children:
            return probs

        actions = np.array(list(root.children.keys()), dtype=np.int64)
        visit_counts = np.array(
            [root.children[action].visit_count for action in actions],
            dtype=np.float32,
        )

        if temperature <= 0.0:
            best_idx = int(np.argmax(visit_counts))
            probs[int(actions[best_idx])] = 1.0
            return probs

        adjusted = np.power(visit_counts, 1.0 / temperature)
        total = float(np.sum(adjusted))

        if total <= 0.0:
            uniform_prob = 1.0 / float(actions.size)
            for action in actions:
                probs[int(action)] = uniform_prob
            return probs

        dist = adjusted / total
        for action, prob in zip(actions, dist, strict=True):
            probs[int(action)] = float(prob)
        return probs

    def get_best_move(self, board: AtaxxBoard) -> tuple[int, int, int, int] | None:
        """
        Return best move by visit count. Returns `None` when pass is best/only action.
        """
        action_probs = self.run(
            board=board,
            add_dirichlet_noise=False,
            temperature=0.0,
        )
        best_action_idx = int(np.argmax(action_probs))
        return ACTION_SPACE.decode(best_action_idx)
