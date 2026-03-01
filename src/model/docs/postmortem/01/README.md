# AtaxxZero — Post-mortem y aprendizajes

> **Resumen:** El modelo entrenó 40 iteraciones y aprendió únicamente a oscilar piezas
> (mover adelante y atrás indefinidamente). Este documento explica por qué pasó,
> cómo se diagnosticó, y qué se aprendió para futuros proyectos de RL con self-play.

---

## Qué se esperaba vs. qué pasó

**Se esperaba:** un modelo que aprenda a jugar Ataxx progresivamente mejor a través
de self-play, guiado por MCTS con 400 simulaciones por movimiento.

**Lo que pasó:** el modelo convergió a una estrategia degenerada — oscilar la misma
pieza entre dos casillas hasta que el contador de `half_moves` llegara a 100 y el
juego terminara en empate. Después de 40 iteraciones y ~70 episodios por iteración,
el comportamiento no cambió.

---

## Por qué MCTS no lo corrigió solo

La pregunta más natural es: *si MCTS hace 400 simulaciones buscando el mejor
movimiento, ¿por qué no encontró que oscilar es malo?*

La respuesta es que **MCTS en AlphaZero no es un buscador independiente de la
verdad — es un amplificador del modelo neuronal**. En cada nodo hoja del árbol,
MCTS le pregunta al modelo dos cosas:

1. **Prior de política:** qué movimientos vale la pena explorar
2. **Estimación de valor:** qué tan buena es esta posición

Si el modelo dice que oscilar tiene valor ~0 (empate) y no conoce nada mejor,
MCTS hace 400 simulaciones que confirman exactamente eso. No tiene criterio propio
para saber que existe algo mejor — solo puede explorar lo que el modelo le señala.

El loop completo:

```
Iter 1:  modelo aleatorio → MCTS elige al azar → juegos terminan en empate → z=0
Iter 2:  modelo aprendió que z≈0 es lo normal → MCTS confirma → más empates → más z=0
Iter 3+: el ciclo se cierra — oscilar es "óptimo" según el modelo
         MCTS con 400 sims lo confirma con alta confianza
         nunca explora alternativas porque el prior no las señala
```

---

## Las causas raíz (en orden de impacto)

### 1. Sin detección de repetición de posición

**Archivo:** `game/board.py`

El tablero tenía un límite de 100 `half_moves` para terminar juegos, pero nunca
detectaba si la misma posición ocurría repetidamente. Oscilar una pieza era
completamente legal e incluso racional: producía un empate (`z=0`) que era
mejor que perder (`z=-1`). Sin penalización real, el comportamiento era
técnicamente óptimo dado lo que el modelo conocía.

**Fix:** agregar un `Counter` de hashes de posición y declarar empate a las
3 repeticiones, igual que en ajedrez.

---

### 2. Cold start: 80% self-play desde la iteración 1

**Archivo:** `train.py` — CONFIG `opponent_self_prob: 0.8`

En la iteración 1, ambos jugadores son modelos aleatorios. Con 80% self-play,
el modelo aprende patrones de un oponente igual de malo. Nadie castiga la
oscilación porque el oponente también oscila. El comportamiento degenerado se
solidifica antes de que exista señal real de entrenamiento.

**Fix:** curriculum dinámico — empezar con 0% self-play y 90% heurístico,
aumentando el self-play gradualmente a medida que el modelo mejora. Agregar
un warmup supervisado de ~600 partidas heurística-vs-heurística antes de
comenzar el loop.

---

### 3. El estado de Adam se reseteaba cada iteración

**Archivo:** `model/system.py` + `train.py`

`PyTorch Lightning` llama `configure_optimizers()` cada vez que se invoca
`trainer.fit()`. Eso recrea el optimizador Adam desde cero — con momentum
`m=0` y varianza `v=0` por parámetro — en cada iteración del loop.

Adam necesita varios pasos para construir estimaciones útiles de sus momentos
internos. Con un reset cada 5 epochs, el optimizador siempre estaba en modo
"frío" y nunca acumulaba momentum útil. En 20 iteraciones × 5 epochs, nunca
alcanzó velocidad de crucero.

**Fix:** usar un `Callback` de PL que guarda y restaura el `state_dict()` del
optimizer entre llamadas a `trainer.fit()`.

---

### 4. Data leakage entre train y validación

**Archivo:** `data/dataset.py`

```python
# Código problemático:
train_dataset = AtaxxDataset(buffer=buffer)        # buffer completo
val_dataset   = ValidationDataset(buffer=buffer)   # últimos 10% del mismo buffer
```

`AtaxxDataset` incluía **todos** los ejemplos del buffer, incluyendo los mismos
que iban a `ValidationDataset`. El modelo veía esos ejemplos tanto en training
como en validación. Consecuencias:

- `val/loss` medía memorización, no generalización
- `ModelCheckpoint(monitor="val/loss")` guardaba el modelo que mejor memorizaba
- Todo lo loggeado en TensorBoard para validación era una métrica inventada

**Fix:** hacer el corte train/val consistente en ambos datasets — `AtaxxDataset`
toma los primeros `(1 - val_split) * N` ejemplos, `ValidationDataset` toma
los últimos `val_split * N`.

---

### 5. Value loss sin coeficiente de balance

**Archivo:** `model/system.py`

```python
# Código problemático:
loss = loss_v + loss_pi   # suma directa
```

`loss_pi` (cross-entropy de política) típicamente vale entre 0.5 y 4.0.
`loss_v` (MSE de valor en rango [-1, 1]) típicamente vale entre 0.05 y 0.4.
El gradiente estaba dominado por la política en una proporción de ~10:1.
El value head recibía muy poca señal y aprendía lento — lo cual impedía
que el modelo estimara correctamente qué posiciones son ganadoras.

**Fix:** `loss = loss_pi + 0.5 * loss_v` — estándar en la literatura AlphaZero.

---

### 6. Train/inference mismatch en action masking

**Archivo:** `model/system.py`

Durante MCTS (inferencia) el modelo recibe una máscara de acciones legales que
pone `-inf` en acciones ilegales antes del softmax. Durante entrenamiento no se
pasaba ninguna máscara — el modelo aprendía a distribuir probabilidad entre
todas las ~2000 acciones posibles incluyendo ilegales. Había un gap sistemático
entre lo que el modelo aprendía y cómo era usado en inferencia.

**Fix:** derivar la máscara legal del `target_pi` (`acciones con prob > 0`)
y pasarla al forward durante `_common_step`.

---

### 7. Red ciega a la fase del juego

**Archivo:** `game/board.py` — `get_observation()`

La observación tenía 3 canales (piezas propias, piezas rivales, casillas vacías)
pero no incluía `half_moves`. La red no podía distinguir el turno 3 del turno 95,
no podía aprender a jugar diferente en apertura vs. final, y no podía detectar
que oscilar en el turno 90 significaba empate inminente.

**Fix:** agregar un canal 4 con `half_moves / 100.0` (valor normalizado entre 0 y 1).

---

### 8. Temperatura caía demasiado pronto

**Archivo:** `train.py` — CONFIG `temp_threshold: 15`

Con `temp_threshold=15`, a partir del turno 16 el modelo era completamente greedy.
Si la política era mala hasta ese punto (inevitable en early training), cada turno
reforzaba determinísticamente la decisión mala. No había exploración suficiente
para escapar del mínimo local.

**Fix:** subir a `temp_threshold: 28`.

---

### 9. Dirichlet alpha fijo y demasiado alto

**Archivo:** `engine/mcts.py`

```python
self._add_dirichlet_noise(root, alpha=0.3, frac=0.25)
```

La fórmula estándar de AlphaZero es `alpha = 10 / N` donde N es el branching
factor típico. En Ataxx 7×7 hay ~60-100 movimientos legales, entonces
`alpha ≈ 0.1-0.17`. Con `alpha=0.3`, el ruido era demasiado uniforme y
aplastaba el prior aprendido, forzando exploración aleatoria de movimientos malos.

**Fix:** `alpha = max(0.03, 10.0 / n_legal)` — adaptativo al número de acciones legales.

---

### 10. Batch MCTS sin virtual loss

**Archivo:** `engine/mcts.py`

Con `leaf_batch_size=8`, las 8 simulaciones del mismo batch podían terminar en
el mismo nodo hoja porque el PUCT score no cambiaba entre simulaciones del mismo
batch. El batch no aportaba diversidad real — era casi equivalente a correr 1
simulación 8 veces. Peor aún, el segundo `_populate_children()` sobre el mismo
nodo sobreescribía los hijos creados por el primero, perdiendo el backprop
de la primera simulación.

**Fix:** aplicar virtual loss durante la selección para que las simulaciones
del mismo batch diverjan hacia nodos distintos.

---

### 11. Scheduler coseno nunca completaba un ciclo

**Archivo:** `model/system.py` + `train.py`

```python
# Inicialización:
system = AtaxxZero(max_epochs=iterations * epochs)  # T_max = 100

# Pero cada trainer solo corre:
trainer = _build_trainer(epochs=5)   # y llama configure_optimizers() de nuevo
```

El scheduler coseno se creaba con `T_max=100` pero se reiniciaba en cada
iteración. El LR siempre estaba en la fase "caliente" del coseno y nunca
annealeaba de verdad.

**Fix:** `max_epochs=epochs` (5) para que cada trainer vea un ciclo completo,
o un solo trainer que persiste entre iteraciones.

---

### 12. Post-LN en lugar de Pre-LN

**Archivo:** `model/transformer.py`

```python
encoder_layer = nn.TransformerEncoderLayer(norm_first=False)  # Post-LN
```

Post-LN (norma después de la atención) tiene problemas de gradientes en capas
tempranas, especialmente con 6 layers y datos ruidosos. Pre-LN (`norm_first=True`)
es más estable y es el estándar en modelos modernos.

**Fix:** `norm_first=True`.

---

## Lecciones generales para proyectos de RL con self-play

### Sobre el loop de entrenamiento

- **El self-play puro desde el inicio es peligroso.** Sin un punto de referencia
  externo (heurística, datos humanos), el modelo puede converger a estrategias
  degeneradas y no saberlo. Siempre empezar con imitación o curriculum.

- **Los bugs silenciosos son los más costosos.** El data leakage y el reset de
  Adam no generaban errores ni warnings — simplemente hacían que el entrenamiento
  fuera menos efectivo de manera invisible durante 40 iteraciones.

- **Val/loss mentiroso es peor que no tener validación.** Si el checkpoint se
  guarda basado en una métrica falsa, el "mejor modelo" guardado puede ser el peor.

### Sobre MCTS + red neuronal

- **MCTS no corrige un modelo malo — lo amplifica.** Si el prior señala hacia
  movimientos malos, 400 simulaciones confirman que son malos con alta confianza.
  La calidad del modelo es el límite superior de la calidad del MCTS.

- **Las reglas del juego deben cerrar todos los loops de comportamiento degenerado.**
  Si oscilar es legal e impune, el modelo lo va a aprender. La regla de repetición
  no es un detalle — es parte del espacio de estados que el modelo necesita para
  razonar correctamente.

- **La observación debe incluir todo lo que el jugador necesita para decidir.**
  Si la fase del juego importa para la estrategia (y en Ataxx importa), `half_moves`
  debe estar en la observación. Un modelo que no ve el reloj no puede aprender
  a usarlo.

### Sobre infraestructura de ML

- **PL + loop manual de self-play requiere cuidado explícito con el estado del
  optimizer.** PyTorch Lightning no está diseñado para llamadas múltiples a
  `trainer.fit()` — hay que manejar el estado manualmente.

- **Policy loss y value loss tienen escalas completamente distintas.** Sumarlos
  directamente sesgará el entrenamiento hacia la pérdida más grande. Siempre
  usar un coeficiente de balance explícito.

- **Train/inference consistency importa.** Si el modelo usa action masking en
  inferencia pero no en entrenamiento, hay un gap sistemático que introduce
  ruido en los gradientes.

---

## El ciclo vicioso completo — diagrama

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Modelo aleatorio (iter 1)                                     │
│         │                                                       │
│         ▼                                                       │
│   MCTS sin prior útil → explora casi al azar                    │
│         │                                                       │
│         ▼                                                       │
│   Juegos terminan en empate por oscilación (half_moves=100)     │
│   Sin regla de repetición → oscilar es óptimo                   │
│         │                                                       │
│         ▼                                                       │
│   Buffer lleno de z=0 (empates por oscilación)                  │
│         │                                                       │
│         ▼                                                       │
│   Entrenamiento con Adam frío + val/loss falso                  │
│   Value head aprende: "todo vale ~0"                            │
│   Policy head aprende: "los movimientos de oscilación son ok"   │
│         │                                                       │
│         ▼                                                       │
│   MCTS iter 2: prior señala oscilaciones                        │
│   400 sims confirman que oscilar ≈ 0 (correcto dado el modelo)  │
│         │                                                       │
│         └──────────────────────────────────────────────────────►│
│                    (loop se repite 40 veces)                    │
└─────────────────────────────────────────────────────────────────┘
```

Cada componente roto contribuyó a cerrar el ciclo. Ninguno solo hubiera causado
el desastre completo — fue la combinación de todos operando simultáneamente.

---

## Archivos modificados y naturaleza del cambio

| Archivo | Tipo de cambio | Razón |
|---------|---------------|-------|
| `game/board.py` | Lógica de reglas | Agregar detección de repetición + canal `half_moves` en observación |
| `data/dataset.py` | Corrección de bug | Eliminar data leakage entre train y val |
| `model/system.py` | Corrección de bug + mejora | Action mask en training, coeficiente de value loss, compatibilidad con PL |
| `model/transformer.py` | Mejora de arquitectura | Pre-LN, capa oculta en policy head |
| `engine/mcts.py` | Corrección de bug + mejora | Virtual loss, Dirichlet adaptativo |
| `train.py` | Rediseño parcial | Curriculum, warmup supervisado, OptimizerStateTransfer, temp_threshold |