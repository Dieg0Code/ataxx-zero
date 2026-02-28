from __future__ import annotations

import math
from collections import OrderedDict
from contextlib import nullcontext

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
        use_amp: bool = True,
        cache_size: int = 20_000,
        leaf_batch_size: int = 8,
    ) -> None:
        self.model = model
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.use_amp = use_amp and self.device.type == "cuda"
        self.cache_size = max(0, int(cache_size))
        self.leaf_batch_size = max(1, int(leaf_batch_size))
        self._inference_cache: OrderedDict[
            tuple[int, bytes],
            tuple[np.ndarray, np.ndarray, float],
        ] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    def run(
        self,
        board: AtaxxBoard,
        add_dirichlet_noise: bool = False,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Run MCTS from `board` and return visit-based policy probabilities.
        """
        probs, _ = self.run_with_root(
            board=board,
            root=None,
            add_dirichlet_noise=add_dirichlet_noise,
            temperature=temperature,
        )
        return probs

    def run_with_root(
        self,
        board: AtaxxBoard,
        root: MCTSNode | None,
        add_dirichlet_noise: bool = False,
        temperature: float = 1.0,
    ) -> tuple[np.ndarray, MCTSNode | None]:
        """
        Run MCTS from `board` starting at an optional cached root.
        Returns (visit_probs, root_after_search).
        """
        self._cache_hits = 0
        self._cache_misses = 0
        probs = np.zeros(ACTION_SPACE.num_actions, dtype=np.float32)
        if board.is_game_over():
            return probs, None

        if root is None:
            root = MCTSNode(prior=1.0)
            self._expand(root, board)
        elif not root.children:
            self._expand(root, board)

        if add_dirichlet_noise and root.children:
            self._add_dirichlet_noise(root, alpha=0.3, frac=0.25)

        sims_done = 0
        while sims_done < self.n_simulations:
            batch_budget = min(self.leaf_batch_size, self.n_simulations - sims_done)
            pending: list[tuple[MCTSNode, AtaxxBoard, list[MCTSNode]]] = []

            for _ in range(batch_budget):
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
                    self._backpropagate(search_path, value)
                else:
                    pending.append((node, scratch_board, search_path))
                sims_done += 1

            if pending:
                leaves = [(node, leaf_board) for node, leaf_board, _ in pending]
                values = self._expand_batch(leaves)
                for (_, _, search_path), value in zip(pending, values, strict=True):
                    self._backpropagate(search_path, value)

        return self._get_action_probs(root, temperature), root

    def advance_root(self, root: MCTSNode | None, action_idx: int) -> MCTSNode | None:
        """Advance cached root after applying `action_idx` on the real board."""
        if root is None:
            return None
        return root.children.get(action_idx)

    def _populate_children(
        self,
        node: MCTSNode,
        legal_action_indices: np.ndarray,
        legal_priors: np.ndarray,
    ) -> None:
        if legal_action_indices.size == 0:
            return
        prior_sum = float(np.sum(legal_priors))
        if prior_sum <= 0.0:
            uniform_prior = 1.0 / float(legal_action_indices.size)
            for action_idx in legal_action_indices:
                node.children[int(action_idx)] = MCTSNode(prior=uniform_prior)
            return
        for action_idx, prior in zip(legal_action_indices, legal_priors, strict=True):
            node.children[int(action_idx)] = MCTSNode(prior=float(prior / prior_sum))

    def _expand_batch(self, leaves: list[tuple[MCTSNode, AtaxxBoard]]) -> list[float]:
        results: list[float] = [0.0] * len(leaves)
        to_infer: list[tuple[int, MCTSNode, np.ndarray, tuple[int, bytes], np.ndarray]] = []

        for idx, (node, board) in enumerate(leaves):
            cache_key = (int(board.current_player), board.grid.tobytes())
            cached = self._inference_cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                legal_action_indices, legal_priors, value = cached
                self._inference_cache.move_to_end(cache_key)
                self._populate_children(node, legal_action_indices, legal_priors)
                results[idx] = value
                continue
            self._cache_misses += 1

            valid_moves = board.get_valid_moves()
            if len(valid_moves) == 0:
                legal_action_indices = np.array([ACTION_SPACE.pass_index], dtype=np.int64)
            else:
                legal_action_indices = np.fromiter(
                    (ACTION_SPACE.encode(move) for move in valid_moves),
                    dtype=np.int64,
                    count=len(valid_moves),
                )
            obs = board.get_observation()
            to_infer.append((idx, node, legal_action_indices, cache_key, obs))

        if not to_infer:
            return results

        states = torch.stack(
            [torch.from_numpy(obs) for _, _, _, _, obs in to_infer],
            dim=0,
        ).to(self.device)
        action_mask = torch.zeros(
            (len(to_infer), ACTION_SPACE.num_actions),
            device=self.device,
            dtype=states.dtype,
        )
        for batch_idx, (_, _, legal_action_indices, _, _) in enumerate(to_infer):
            legal_idx_tensor = torch.from_numpy(legal_action_indices).to(self.device)
            action_mask[batch_idx, legal_idx_tensor] = 1.0

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), amp_ctx:
            policy_logits, value_tensor = self.model(states, action_mask=action_mask)

        values_np = value_tensor.squeeze(1).detach().float().cpu().numpy()
        for batch_idx, (idx, node, legal_action_indices, cache_key, _) in enumerate(to_infer):
            legal_idx_tensor = torch.from_numpy(legal_action_indices).to(self.device)
            legal_logits = policy_logits[batch_idx].index_select(0, legal_idx_tensor)
            legal_priors = torch.softmax(legal_logits, dim=0).float().cpu().numpy()
            value = float(values_np[batch_idx])
            self._populate_children(node, legal_action_indices, legal_priors)
            results[idx] = value
            if self.cache_size > 0:
                self._inference_cache[cache_key] = (
                    legal_action_indices.copy(),
                    legal_priors.copy(),
                    value,
                )
                if len(self._inference_cache) > self.cache_size:
                    self._inference_cache.popitem(last=False)

        return results

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
        return self._expand_batch([(node, board)])[0]

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
            probs[actions] = uniform_prob
            return probs

        dist = adjusted / total
        probs[actions] = dist
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

    def cache_stats(self) -> dict[str, float | int]:
        total = self._cache_hits + self._cache_misses
        hit_rate = (float(self._cache_hits) / float(total)) if total > 0 else 0.0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_entries": len(self._inference_cache),
        }
