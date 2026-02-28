# Fase 1 - Bots como Usuarios Persistentes (sin invitaciones)

## Objetivo
Permitir que `easy`, `normal`, `hard` y `model` sean cuentas reales de usuario (`is_bot=true`) para jugar contra humanos usando el flujo actual de partidas, sin introducir todavía sistema de invitaciones.

Resultado esperado:
- El humano crea partida contra un bot-user.
- El backend resuelve el perfil del bot y ejecuta su jugada automáticamente cuando toque su turno.
- Replay/metadata reflejan correctamente rival, agente y modo.

---

## Estado actual (ya existe)
- `User` tiene soporte de bots: `is_bot`, `bot_kind`, `model_version_id`, `is_hidden_bot`.
- `Game` soporta `player1_id/player2_id` y `player1_agent/player2_agent`.
- Endpoints de partida: `/api/v1/matches` y `/api/v1/gameplay/games`.
- No existe sistema de invitaciones/challenges.

---

## Diseño funcional de Fase 1

### 1) Catálogo de bots oficiales
Crear y mantener 4 usuarios bot (visibles o no según producto):
- `ub_ai_easy`
- `ub_ai_normal`
- `ub_ai_hard`
- `ub_ai_model`

Todos con:
- `is_bot=true`
- `is_active=true`
- `is_admin=false`
- `bot_kind`:
  - heurísticos: `heuristic`
  - modelo: `model`
- `model_version_id` solo para `ub_ai_model` (version activa o fija)

### 2) Config de comportamiento por bot
Agregar tabla/config de perfil para no inferir por username.

`bot_profile` (nueva tabla recomendada):
- `user_id` (PK/FK a user)
- `agent_type` (`heuristic` | `model`)
- `heuristic_level` (`easy` | `normal` | `hard` | null)
- `model_mode` (`fast` | `strong` | null)
- `enabled` (bool)
- `created_at`, `updated_at`

Regla:
- si `agent_type=heuristic` => usar `heuristic_level`
- si `agent_type=model` => usar `model_mode` + `model_version_id` del user o activa global

### 3) Creación de partida contra bot
Para fase 1, no hay invitación:
- Cliente crea partida directo con `player2_id=bot_user_id`.
- Backend valida que `player2_id` bot existe y está habilitado.
- Backend fija `player2_agent` automáticamente según `bot_profile`.

### 4) Turno del bot (server-authoritative)
En `matches`:
- Si el turno pertenece a bot, endpoint de movimiento humano procesa la jugada humana.
- Luego backend calcula y persiste jugada del bot en la misma operación o en endpoint dedicado de "advance".

Recomendación inicial (simple y robusta):
- Nuevo endpoint: `POST /api/v1/matches/{game_id}/advance-bot`
- Solo callable por participante humano o backend trusted.
- Ejecuta 1 jugada bot (o pass legal) y persiste `mode` correcto.

Luego opcional:
- auto-advance dentro de `submit_move` (humano mueve y bot responde en la misma request).

### 5) Replay y metadata
- `GameMove.mode` debe guardar modo real:
  - `heuristic_easy|normal|hard`
  - `fast|strong`
  - `manual` para humano
- `GameDetail` muestra "Modo IA detectado" desde moves reales.

---

## Contratos API (fase 1)

### A) Listar bots jugables
`GET /api/v1/identity/bots`

Response:
```json
{
  "items": [
    {
      "id": "uuid",
      "username": "ub_ai_normal",
      "bot_kind": "heuristic",
      "agent_type": "heuristic",
      "heuristic_level": "normal",
      "model_mode": null,
      "enabled": true
    }
  ]
}
```

### B) Crear partida vs bot
Usar endpoint existente `POST /api/v1/matches` (o `gameplay/games`) con:
- `player2_id = bot_user_id`
- backend ignora intentos de setear agentes inconsistentes y deriva desde `bot_profile`.

### C) Avanzar turno bot
`POST /api/v1/matches/{game_id}/advance-bot`

Response:
```json
{
  "applied": true,
  "move": {
    "ply": 7,
    "player_side": "p2",
    "mode": "heuristic_normal"
  },
  "game_status": "in_progress"
}
```

---

## Cambios por módulo

### DB / Models
- Nuevo: `src/api/db/models/bot_profile.py`
- Export en `src/api/db/models/__init__.py`
- Alembic migration para `bot_profile` + índices + FK

### Identity
- `src/api/modules/identity/`
  - endpoint para listar bots activos
  - opcional endpoint admin para upsert bot profile

### Matches
- `src/api/modules/matches/service.py`
  - resolver `bot_profile` por `player_side` actual
  - construir jugada bot con:
    - `heuristic_move(...)` o
    - `inference_service.predict(..., mode)`
  - persistir move con `mode` real
- `src/api/modules/matches/router.py`
  - `POST /matches/{game_id}/advance-bot`

### Gameplay (compat)
- Mantener consistente persistencia de `mode` en replays.

### Frontend web
- Match screen:
  - selector de rival desde `GET /identity/bots`
  - al finalizar jugada humana: llamar `advance-bot`
- Replay:
  - sin fallback engañoso de modo IA

---

## Reglas de negocio
- No permitir `rated=true` contra bot en fase 1.
- Bot inactivo o disabled -> 400/409 al crear partida.
- Solo participantes o admin pueden avanzar bot.
- Si bot sin jugadas legales y pass legal -> registrar pass.

---

## TDD plan (orden recomendado)

1. **Unit tests bot profile resolution**
- archivo sugerido: `tests/test_api_matches_bot_profile.py`
- casos:
  - heuristic normal -> selecciona `heuristic_normal`
  - model fast -> selecciona `fast`
  - bot disabled -> error

2. **Integration test create vs bot**
- `POST /matches` con `player2_id` bot
- assert `player2_agent` derivado correctamente

3. **Integration test advance-bot**
- crea partida vs bot
- humano mueve
- `POST /advance-bot`
- assert nuevo move p2 con `mode` correcto

4. **Replay test mode correctness**
- obtener replay
- assert `moves[].mode` contiene `heuristic_normal` o `fast/strong` real

5. **Frontend test**
- `MatchPage.test.tsx`
- mock de `/identity/bots` + `/advance-bot`
- assert flujo humano->bot

---

## Rollout seguro
- Feature flag: `ENABLE_BOT_USERS_V1=true`
- Si flag off, usar flujo actual.
- Crear script idempotente de seed bots.

Script sugerido:
- `scripts/seed_bot_users.py`
- crea/actualiza bots + perfiles sin duplicar.

---

## No incluido en Fase 1
- Invitaciones/challenges (pendiente Fase 2)
- Matchmaking en cola
- ELO ranked contra bots

---

## Checklist de implementación
- [ ] Modelo `bot_profile` + migración
- [ ] Servicio/listado de bots
- [ ] Crear partida vs bot (derivación de agente)
- [ ] Endpoint `advance-bot`
- [ ] Replay con `mode` correcto
- [ ] Tests backend (unit + integration)
- [ ] Integración web (selector bot + llamada advance)
- [ ] Seed script bots
- [ ] Documentación API
