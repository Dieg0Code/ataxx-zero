# AtaxxZero - Post-mortem 02: desempate determinista en MCTS

> **Resumen:** El modelo no solo era debil; el search tambien estaba resolviendo
> empates de manera arbitraria pero sistematica. Cuando la policy de apertura era
> plana, MCTS elegia siempre la primera jugada legal segun el orden interno del
> action space. Eso contaminaba self-play, eval, comparaciones entre checkpoints
> y juego local. El fix no agrega conocimiento externo: solo elimina un sesgo de
> implementacion.

---

## Que se esperaba vs. que paso

**Se esperaba:** si varias jugadas iniciales eran equivalentes para el modelo,
MCTS debia tratarlas como equivalentes y la partida debia variar entre lineas
de apertura razonables.

**Lo que paso:** la apertura colapsaba a una sola linea repetida. En pruebas
checkpoint-vs-checkpoint y hasta en `modelo vs si mismo`, el lado azul ganaba
una y otra vez porque el lado rojo caia siempre en la misma secuencia mala.

El problema se veia asi:

1. `p1`: `(0, 0) -> (0, 1)`
2. `p2`: `(0, 6) -> (0, 4)`
3. `p1`: `(0, 0) -> (0, 2)`
4. `p2`: `(0, 4) -> (0, 3)`

Desde ahi, rojo repetia una continuacion pobre y azul la castigaba casi siempre.

---

## Diagnostico correcto

La causa principal **no** era ausencia de augmentacion por simetrias.

Ese diagnostico era insuficiente porque el repo ya tenia transformaciones de
rotacion/reflexion para observacion y policy en `src/data/dataset.py`.

La causa real fue esta combinacion:

1. la red daba una policy inicial casi uniforme;
2. `get_valid_moves()` devolvia las jugadas en un orden fijo;
3. MCTS usaba desempates deterministas en seleccion y decision final;
4. el primer elemento del action space quedaba privilegiado aunque el search no
   hubiera encontrado una razon real para preferirlo.

---

## Evidencia concreta

### 1. La policy de apertura estaba casi plana

Antes del search, en la posicion inicial, tanto `v2_093` como `v4_125` daban
probabilidades casi iguales a las aperturas legales. No habia una preferencia
fuerte aprendida por el modelo.

Ejemplo observado:

- varias jugadas legales con `~0.0625`
- valor cercano a `0`

Eso significa: la red todavia no sabia abrir, pero tampoco estaba obligando una
jugada unica por si sola.

### 2. MCTS convertia esa policy plana en una decision one-hot fija

Con el codigo original, al correr MCTS desde la posicion inicial:

- sin ruido;
- con `temperature=0.0`;
- y con muchos empates exactos o numericamente equivalentes,

el resultado terminaba siendo:

- una sola jugada con probabilidad `1.0`
- todas las demas con `0.0`

En otras palabras: el search no estaba "descubriendo" una gran apertura;
simplemente estaba cristalizando el primer empate.

### 3. El sesgo aparecia incluso en auto-duelo

Cuando un checkpoint jugaba contra si mismo:

- antes del fix, seguia apareciendo el patron "siempre gana azul";
- por lo tanto, el problema no era que un checkpoint fuera mejor que otro;
- el problema estaba en el proceso de decision.

### 4. Despues del fix, la apertura dejo de colapsar

Despues del cambio:

- `v4_125` abrio con 6 primeras jugadas distintas en 10 semillas;
- `v2_093` abrio con 7 primeras jugadas distintas en 10 semillas.

Eso era exactamente lo que faltaba: diversidad cuando el search no tiene una
preferencia legitima.

---

## La causa raiz en codigo

### 1. Orden fijo de acciones legales

**Archivo:** `src/game/board.py`

`get_valid_moves()` genera movimientos siguiendo el orden de recorrido del
tablero y del radio de movimientos.

Eso esta bien por si solo, pero significa que existe un "primer movimiento"
estable para cualquier posicion.

### 2. Insercion de hijos en el mismo orden

**Archivo:** `src/engine/mcts.py`

`_populate_children()` insertaba los nodos hijo siguiendo ese orden fijo de
acciones legales.

### 3. `_select_child()` rompia empates quedandose con el primero

**Archivo:** `src/engine/mcts.py`

La version anterior hacia esto conceptualmente:

```python
if score > best_score:
    best_score = score
    best_child = child
```

Si dos hijos tenian el mismo `score`, el segundo nunca reemplazaba al primero.
Eso sesgaba la seleccion hacia el primer hijo insertado.

### 4. `temperature=0.0` tambien rompia empates de forma fija

**Archivo:** `src/engine/mcts.py`

Cuando varias acciones terminaban con el mismo `visit_count`, `_get_action_probs`
usaba `argmax`, que otra vez se queda con la primera.

### 5. Inference/eval/comparison heredaban ese sesgo

**Archivos:**

- `src/agents/model_agent.py`
- `src/training/eval_runtime.py`
- `scripts/compare_checkpoints.py`

Todos usaban:

- `add_dirichlet_noise=False`
- `temperature=0.0`
- `argmax`

Eso no es incorrecto cuando el search tiene una preferencia clara.
Se vuelve patologico cuando el root esta lleno de empates.

---

## Por que esto si era un bug de comportamiento

No era "solo una decision de diseno".

Un desempate sistematico basado en orden de enumeracion:

- no representa conocimiento del juego;
- no representa preferencia aprendida por la red;
- no representa una conclusion fuerte de MCTS.

Representa solo un accidente de implementacion.

En problemas de search, dejar empates ligados al orden interno del action space
es peligroso porque:

- sesga self-play;
- reduce diversidad del buffer;
- falsifica comparaciones entre modelos;
- puede esconder o exagerar fortalezas que no existen.

---

## El fix aplicado

**Archivo:** `src/engine/mcts.py`

Se hicieron dos cambios:

### 1. Desempate neutral en `_select_child()`

Cuando varios hijos tienen el mismo `score` dentro de una tolerancia numerica
pequena, ahora se elige uno al azar entre los empatados.

### 2. Desempate neutral en `_get_action_probs(..., temperature=0.0)`

Cuando varias acciones tienen el mismo maximo de visitas, ya no se toma siempre
el primer `argmax`. Se elige aleatoriamente entre las empatadas.

Importante: esto **no** agrega libro de aperturas, heuristicas nuevas ni reglas
externas. Solo evita que el action space imponga una apertura arbitraria.

---

## Por que el fix es consistente con el espiritu de self-play

Una objecion razonable seria:

> "Si el modelo aprende solo, por que tocar el search?"

Respuesta:

- no se altero la evaluacion del modelo;
- no se metio conocimiento humano;
- no se forzo ninguna apertura;
- solo se corrigio un sesgo artificial de implementacion.

Si dos acciones estan empatadas para el search, **cualquiera** de ellas es
coherente. Lo incoherente era privilegiar siempre la primera por su indice.

---

## Efecto observado despues del fix

### Lo que mejoro

- el modelo ya no abre siempre igual;
- las comparaciones checkpoint-vs-checkpoint dejaron de ser trivialmente
  "siempre gana azul";
- el comportamiento local jugando contra el modelo se ve menos mecanico.

### Lo que no mejoro automaticamente

- el modelo sigue siendo debil;
- `v4_125` no supero de forma util a `v2_093`;
- siguen apareciendo muchas partidas largas o tacticamente pobres.

Eso confirma algo importante:

> El desempate determinista estaba empeorando el sistema, pero no era la unica
> causa de debilidad del modelo.

---

## Riesgos futuros encontrados durante la revision

### 1. `cache_hit_rate` puede inducir a error

**Archivo:** `src/engine/mcts.py`

Los contadores de cache se resetean por corrida de `run_with_root()`, asi que el
resumen agregado de self-play no representa bien todo el trabajo de una iteracion.

No causo el bug de apertura, pero si puede llevar a diagnosticos falsos.

### 2. Los workers de self-play dependen de `CONFIG` global con `spawn`

**Archivo:** `src/training/selfplay_runtime.py`

Los procesos hijos leen `cfg_*` desde un global mutable. Eso hoy no parece haber
causado este problema, pero es una fragilidad real para futuros cambios de
configuracion.

### 3. Los campos `opponent_*` del JSON no reflejan el curriculum real

**Archivos:**

- `src/training/config_runtime.py`
- `src/training/curriculum.py`

La mezcla efectiva de self-play la domina `get_curriculum_mix()`, no los campos
legacy del JSON de config. Eso puede confundir al leer metadata de una corrida.

---

## Lecciones

### Sobre search

- Los empates en MCTS no deben resolverse por orden interno del action space.
- Un search que parece "determinista y fuerte" puede estar solo repitiendo un
  sesgo de enumeracion.
- Cuando el modelo esta verde, los desempates importan mucho mas que cuando el
  modelo ya es fuerte.

### Sobre diagnostico

- Antes de culpar a la arquitectura o a la ausencia de simetrias, hay que revisar
  el camino completo de decision:
  - policy cruda
  - expansion de hijos
  - seleccion PUCT
  - decision final
- Una comparacion de checkpoints puede ser inutil si ambos modelos estan jugando
  una apertura artificialmente fijada por el search.

### Sobre entrenamiento

- Arreglar este bug era necesario, pero no suficiente.
- Despues del fix, lo que queda visible es la calidad real del modelo.
- Eso permite que el siguiente experimento de training sea una medicion mas
  limpia, no una mezcla de modelo debil + search sesgado.

---

## Estado final

Despues de este post-mortem, la conclusion operativa es:

1. el fix de MCTS debe quedarse;
2. `policy_spatial_v2_iter_093` sigue siendo el bootstrap mas seguro;
3. el siguiente rerun de Kaggle debe hacerse con este fix ya incluido;
4. si el modelo sigue malo despues de eso, el siguiente cuello de botella ya no
   es el desempate de apertura sino la calidad del aprendizaje.

