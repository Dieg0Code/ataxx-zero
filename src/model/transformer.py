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

        self.input_proj = nn.Linear(3, d_model)
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

        self._init_weights()

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
            x: [batch, 3, 7, 7]
            action_mask: Optional [batch, num_actions] with 1.0 for legal actions.
        Returns:
            policy_logits: [batch, num_actions]
            value: [batch, 1]
        """
        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1).contiguous().view(batch_size, self.num_cells, 3)
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
