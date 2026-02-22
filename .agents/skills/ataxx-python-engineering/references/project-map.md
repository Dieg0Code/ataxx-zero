# Project Map

## Main modules

- `src/game`: board state, rules, action-space, serialization.
- `src/engine`: search logic (`MCTS`).
- `src/model`: neural net + lightning training system.
- `src/inference`: checkpoint-backed move service.
- `src/agents`: heuristic/random/model move policies.
- `src/ui`: pygame arena rendering/effects/layout/theme.
- `src/api`: backend API, settings, db session/models.
- `tests`: numerics, rules, serialization, inference, agents.

## Known fragile areas

- Legal move/pass invariants in `AtaxxBoard.step`.
- Action encode/decode consistency (`ACTION_SPACE`).
- Inference checkpoint loading (`.pt` vs `.ckpt`) and mode handling.
- Script import-path bootstrap for `scripts/play_pygame.py`.
- Data/typing boundaries in UI particle structures.

## Preferred boundaries

- Keep game rules independent from UI/API.
- Keep inference independent from training loop internals.
- Keep API orchestration outside domain logic.
