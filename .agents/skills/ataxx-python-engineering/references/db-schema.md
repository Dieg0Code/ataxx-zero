# DB Schema (Current + Planned)

## Current implemented tables

### `user`
- Identity and profile.
- Bot support:
  - `is_bot`
  - `bot_kind`
  - `is_hidden_bot`
  - `model_version_id`

### `modelversion`
- Model artifact/version metadata:
  - `name`
  - `hf_repo_id`
  - `hf_revision`
  - `checkpoint_uri`
  - `onnx_uri`
  - `is_active`

### `game`
- Match metadata:
  - queue/status/rated
  - players and agent types
  - winner and termination info
  - training flags (`source`, `quality_score`, `is_training_eligible`)

### `gamemove`
- Persisted move history:
  - `game_id`, `ply`, `player_side`
  - move coords (`r1,c1,r2,c2`) or pass (`null`)
  - inference metadata (`mode`, `action_idx`, `value`)
  - replay states (`board_before`, `board_after`)

## Planned tables

### `season`
- Time-bounded ranking context (`starts_at`, `ends_at`, `is_active`).

### `rating`
- User rating snapshot per season.
- Unique `(user_id, season_id)`.

### `rating_event`
- Rating audit record per rated game (`before/after/delta`).

### `leaderboard_entry`
- Materialized/cached ranking rows per season (`rank`, `rating`, stats).

### `training_sample`
- Curated ML samples from self-play/human/mixed data:
  - observation
  - target_policy
  - target_value
  - split (`train|val|test`)

## Business rules

- Bots are first-class users in DB.
- Public UI may hide bot identity, but backend keeps explicit bot flags.
- Only finished + rated games should create rating events.
- Training eligibility is explicit, never implicit.
