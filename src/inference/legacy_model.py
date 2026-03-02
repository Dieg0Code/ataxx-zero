from __future__ import annotations

import torch
import torch.nn as nn

from game.actions import ACTION_SPACE
from game.constants import BOARD_SIZE


class LegacyAtaxxTransformerNet(nn.Module):
    """Transformer legacy (3 canales + policy flatten) para checkpoints historicos."""

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
        self.num_input_channels = 3

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

        self.policy_head = nn.Sequential(
            nn.LayerNorm(d_model * self.num_cells),
            nn.Linear(d_model * self.num_cells, self.num_actions),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        x: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        board_out = encoded[:, 1:].contiguous().view(batch_size, -1)
        policy_logits = self.policy_head(board_out)
        if action_mask is not None:
            min_value = torch.finfo(policy_logits.dtype).min
            policy_logits = policy_logits.masked_fill(action_mask <= 0, min_value)

        value = self.value_head(cls_out)
        return policy_logits, value


class LegacyAtaxxSystem(nn.Module):
    """Wrapper compatible con state_dicts `model.*` de checkpoints legacy."""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model = LegacyAtaxxTransformerNet(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

