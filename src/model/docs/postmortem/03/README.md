# AtaxxZero - Post-mortem 03: por que los modelos no mejoraban y por que `v6` si despego

> **Resumen:** durante varias semanas el entrenamiento parecia "estable" pero no
> producia checkpoints claramente mejores. El problema no era una sola cosa ni
> se resolvia eligiendo otro bootstrap. Habia varias fugas de calidad en la
> generacion de datos, en la representacion del estado y en la integracion del
> search con el training. `policy_spatial_v6` fue la primera corrida que junto
> suficientes fixes correctos como para producir una mejora real.

---

## Sintoma

Los checkpoints anteriores mostraban uno o varios de estos patrones:

- `eval_composite` clavado en `0.0`
- head-to-head contaminado por lineas de apertura degeneradas
- modelos que "se movian" pero no parecian entender el juego
- mejoras marginales entre iteraciones, pero sin salto real de fuerza

El punto importante es este:

> El pipeline corria, pero el modelo estaba aprendiendo sobre datos de calidad
> insuficiente o inconsistente.

---

## Diagnostico correcto

El error operativo durante un tiempo fue leer el problema como:

- "faltan mas iteraciones"
- "falta otro bootstrap"
- "hay que empezar de cero"

Esas hipotesis eran incompletas.

La causa real era acumulativa: el sistema tenia varios defectos pequenos o
medianos que, juntos, degradaban justo la parte mas importante del proyecto:

1. la calidad del self-play;
2. la coherencia entre estado, reward e inferencia;
3. la utilidad de heuristicas y checkpoints como teachers.

---

## Causas raiz

### 1. Cache de inferencia de MCTS incoherente con la observacion real

**Problema**

La cache de inferencia de `MCTS` estaba indexada por `grid` y jugador actual,
pero la red tambien veia `half_moves`.

Eso permitia reutilizar policy/value entre estados distintos desde el punto de
vista del modelo.

**Efecto**

- priors stale dentro del search
- evaluacion inconsistente de estados similares pero no equivalentes
- contaminacion de self-play, eval y comparacion de checkpoints

**Fix**

La cache se paso a construir desde la observacion real que entra a la red.

---

### 2. La red no veia todo el estado que define draws por repeticion

**Problema**

El juego podia terminar por repeticion y el reward shaping tambien castigaba
forced draws, pero esa informacion no estaba en la observacion.

Habia estados con el mismo input para la red que podian diferir en:

- `is_game_over()`
- `is_forced_draw()`

**Efecto**

El target de valor no era Markoviano en la zona donde el modelo mas se atascaba:
los loops.

**Fix**

Se agrego un canal de presion de repeticion y tambien se corrigio la
serializacion para preservar `_position_counts`.

---

### 3. Las heuristicas fuertes castigaban, pero casi no enseñaban policy

**Problema**

En partidas contra heuristicas, el buffer guardaba sobre todo los turnos del
modelo. Cuando movia el rival heuristico, su jugada normalmente no entraba como
target de policy.

**Efecto**

`hard`, `apex`, `gambit` y `sentinel` funcionaban mas como castigo para el
value head que como teacher real para la policy.

**Fix**

Las jugadas de heuristica pasaron a guardarse como ejemplos supervisados en el
buffer.

---

### 4. Training e inferencia no usaban el mismo espacio de decision

**Problema**

En inferencia/MCTS se aplicaba mascara legal. En training, no.

La red tenia que gastar capacidad en aprender "que acciones son ilegales" en
vez de concentrarse en ordenar bien las legales.

**Efecto**

- peor eficiencia del trunk y la policy head
- mismatch entre lo que la red aprende y como se usa en runtime

**Fix**

Training paso a construir la legal-action mask desde el board y a usarla en el
forward igual que MCTS.

---

### 5. Self-play paralelo estaba cayendo a CPU cuando mas importaba

**Problema**

En configuraciones cortas de Kaggle, la generacion de experiencia estaba
corriendo peor de lo necesario porque los workers de self-play terminaban en CPU.

**Efecto**

El sistema estaba optimizando mas el fit que la calidad/cantidad de experiencia.
Eso es la direccion equivocada para AlphaZero-style training.

**Fix**

Se cambio la politica de devices:

- `CUDA x1`: sin pool paralelo, self-play secuencial en GPU
- `CUDA xN`: workers repartidos por GPU
- `CPU`: pool normal en CPU

---

### 6. El sampler de ejemplos recientes repetia demasiado

**Problema**

`sample_recent_mix()` usaba reemplazo de manera agresiva.

**Efecto**

Aunque el buffer fuera grande, el set efectivo de entrenamiento repetia
demasiado ejemplos recientes y estrechaba la diversidad.

**Fix**

El sampler ahora evita reemplazo cuando puede y reparte repeticiones de manera
mucho menos degenerada cuando no queda otra.

---

### 7. Los oponentes checkpoint tampoco estaban enseñando policy

**Problema**

Cuando el rival era otro checkpoint del pool, su politica tampoco se estaba
aprovechando del todo como teacher.

**Efecto**

La liga interna y el pool de checkpoints aportaban oposicion, pero no tanta
senal de imitacion como podian.

**Fix**

Los turnos del checkpoint pasaron a guardarse como targets de policy y, ademas,
se agrego temperatura temprana para que esas aperturas no fueran demasiado
rigidas.

---

### 8. La observacion era demasiado pobre para la estructura real de Ataxx

**Problema**

La red veia piezas, vacios, progreso y repeticion, pero seguia teniendo que
inferir desde cero demasiada estructura del espacio tactico.

**Efecto**

Aprendizaje mas lento de:

- movilidad
- distincion clone/jump
- actividad real de piezas
- cierre y bloqueo de posiciones

**Fix**

La observacion se expandio a 11 canales:

- piezas propias
- piezas rivales
- vacias
- progreso de `half_moves`
- presion de repeticion
- destinos legales de clone propios
- destinos legales de jump propios
- destinos legales de clone rivales
- destinos legales de jump rivales
- piezas propias activas
- piezas rivales activas

Esto no mete libro de aperturas ni heuristicas duras. Solo le da a la red una
representacion mas fiel y util del estado.

---

### 9. La liga interna fallaba al guardar resultados de eval

**Problema**

La liga esperaba resumentes estilo duelo:

- `checkpoint_a_wins`
- `checkpoint_b_wins`

Pero `evaluate_model()` producia:

- `wins`
- `losses`
- `draws`

**Efecto**

La corrida no se caia, pero salia el warning:

`league update failed, continuing training: 'checkpoint_a_wins'`

**Fix**

`record_checkpoint_in_league()` ahora normaliza ambos formatos de resumen antes
de llamar a la liga Elo.

---

## Lo que no era la causa principal

### "Solo faltaban mas iteraciones"

No. Varias corridas anteriores ya estaban entrenando sobre datos suboptimos o
inconsistentes. Mas iteraciones sobre eso solo reforzaban una politica mediocre.

### "Solo habia que cambiar bootstrap"

Tampoco. El bootstrap importaba, pero no era el cuello de botella dominante.
El problema principal era la calidad del loop de aprendizaje.

### "Solo hacia falta otra heuristica"

No por si sola. El problema era que las heuristicas no estaban dejando suficiente
senal de policy en el buffer.

---

## Por que `v6` si mejoro

`policy_spatial_v6_iter_180` fue la primera corrida que combino suficientes fixes
correctos a la vez:

- observacion mas rica y coherente
- search sin cache incoherente
- repeticion visible para la red
- heuristicas y checkpoints funcionando como teachers reales
- training alineado con legal-action mask
- mejor uso de GPU para generar experiencia
- sampler menos degenerado

El resultado ya no fue solo "se mueve distinto". Hubo mejora medible.

Benchmarks locales posteriores al run:

- `v6_180 vs v2_093`: `40-0`
- `v6_180 vs v4_135`: `24-0`
- gauntlet `v6_180`:
  - vs `hard`: `9-3`
  - vs `apex`: `4-8`
  - vs `sentinel`: `8-4`

Y el checkpoint final quedo con:

- `iteration = 180`
- `best_eval_score = 0.8055555555555555`

Eso ya es una mejora real, no una ilusion de logs.

---

## Lecciones

### 1. En AlphaZero, "que el loop corra" no significa "que el loop sirva"

Un pipeline puede:

- generar checkpoints
- subir a HF
- mostrar losses normales

y aun asi estar entrenando sobre una senal mediocre.

### 2. Los bugs de integracion pesan tanto como los hiperparametros

En este caso, varios de los mayores bloqueos no eran:

- learning rate
- tamano de modelo
- numero de iteraciones

Eran problemas de integracion entre:

- board state
- observacion
- MCTS
- replay buffer
- training loop

### 3. Las heuristicas deben enseñar, no solo castigar

Si el rival fuerte solo te gana pero no deja policy targets utiles, el
aprendizaje de la policy avanza mucho mas lento.

### 4. El modelo necesita ver estado estructural, no solo piezas

Movilidad, actividad de piezas y distincion clone/jump resultaron ser features
de alto valor sin romper el espiritu de self-play.

---

## Estado final

La conclusion operativa despues de este post-mortem es:

1. el estancamiento anterior no fue un misterio ni un tema de "mala suerte";
2. habia causas tecnicas concretas y acumulativas;
3. ya se corrigieron las mas importantes;
4. `v6` confirma que el sistema, con esas correcciones, si puede mejorar;
5. los siguientes pasos deben enfocarse en escalar bien ese progreso, no en
   volver a discutir eternamente bootstrap vs. reset total.

