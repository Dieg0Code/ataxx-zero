from __future__ import annotations

import torch
import torch.nn as nn

from game.actions import ACTION_SPACE
from game.constants import BOARD_SIZE


class AtaxxTransformerNet(nn.Module):
    """Transformer policy-value network for Ataxx."""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.board_size = BOARD_SIZE
        self.num_cells = self.board_size * self.board_size
        self.num_actions = ACTION_SPACE.num_actions
        self.num_input_channels = 4

        self.input_proj = nn.Linear(self.num_input_channels, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_cells + 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.policy_src_proj = nn.Linear(d_model, d_model // 2)
        self.policy_dst_proj = nn.Linear(d_model, d_model // 2)
        self.policy_scorer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Tanh(),
        )

        self._build_action_cell_indices()
        self._init_weights()

    def _build_action_cell_indices(self) -> None:
        """Precompute board-token indices (src/dst) for each action in ACTION_SPACE."""
        src_list: list[int] = []
        dst_list: list[int] = []
        for action_idx in range(self.num_actions):
            move = ACTION_SPACE.decode(action_idx)
            if move is None:
                # Pass has no squares; we use index 0 and rely on legal-action masking.
                src_list.append(0)
                dst_list.append(0)
                continue
            r1, c1, r2, c2 = move
            src_list.append(r1 * self.board_size + c1)
            dst_list.append(r2 * self.board_size + c2)

        self.register_buffer("_action_src_idx", torch.tensor(src_list, dtype=torch.long))
        self.register_buffer("_action_dst_idx", torch.tensor(dst_list, dtype=torch.long))

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, 4, 7, 7]
            action_mask: Optional [batch, num_actions] with 1.0 for legal actions.
        Returns:
            policy_logits: [batch, num_actions]
            value: [batch, 1]
        """
        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1).contiguous().view(
            batch_size,
            self.num_cells,
            self.num_input_channels,
        )
        x = self.input_proj(x)

        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, x], dim=1) + self.pos_embed
        encoded = self.encoder(tokens)

        cls_out = encoded[:, 0]
        board_tokens = encoded[:, 1:]

        # Score each move from the semantics of its origin and destination cells.
        src_tokens = board_tokens[:, self._action_src_idx]
        dst_tokens = board_tokens[:, self._action_dst_idx]
        src_feat = self.policy_src_proj(src_tokens)
        dst_feat = self.policy_dst_proj(dst_tokens)
        combined = torch.cat([src_feat, dst_feat], dim=-1)
        policy_logits = self.policy_scorer(combined).squeeze(-1)

        if action_mask is not None:
            min_value = torch.finfo(policy_logits.dtype).min
            policy_logits = policy_logits.masked_fill(action_mask <= 0, min_value)

        value = self.value_head(cls_out)
        return policy_logits, value

    def predict(
        self,
        x: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference helper with softmaxed policy."""
        self.eval()
        with torch.no_grad():
            policy_logits, value = self.forward(x, action_mask=action_mask)
            policy = torch.softmax(policy_logits, dim=1)
        return policy, value
