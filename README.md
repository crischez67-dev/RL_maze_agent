# RL Maze Agent

Proyecto de aprendizaje por refuerzo para entrenar un agente autónomo capaz de navegar en laberintos usando Deep Q-Learning.

El objetivo principal fue construir, entrenar y evaluar un agente que pasara de actuar aleatoriamente a aprender políticas óptimas de navegación en distintos mapas.

---

## 1. Objetivo general

Desarrollar un agente autónomo capaz de aprender a navegar en un laberinto mediante aprendizaje por refuerzo, pasando desde un agente aleatorio de referencia hasta un agente basado en Deep Q-Learning, evaluando su desempeño mediante métricas de recompensa, colisiones, pasos, éxito y visualización del comportamiento aprendido.

---

## 2. Objetivos específicos

- Construir un entorno de laberinto tipo Gym.
- Representar el estado mediante sensores de paredes y posición relativa a la meta.
- Implementar visualización del entorno con Pygame.
- Crear un agente aleatorio como línea base.
- Implementar una red neuronal Q.
- Construir un agente DQN con política epsilon-greedy.
- Implementar Replay Buffer.
- Entrenar el agente usando batches de experiencias.
- Agregar Target Network para mejorar estabilidad.
- Evaluar el desempeño en mapas simples, medianos y múltiples mapas.
- Analizar la capacidad de generalización del agente.
- Comparar resultados mediante métricas y gráficas.

---

## 3. Estructura del proyecto

```text
rl_maze_agent/
│
├── agents/
│   ├── random_agent.py
│   └── dqn_agent.py
│
├── env/
│   ├── maps.py
│   ├── maze_env.py
│   └── renderer.py
│
├── experiments/
│   ├── evaluate_random_agent.py
│   ├── train_dqn_agent.py
│   ├── train_dqn_medium_maze.py
│   ├── train_dqn_multi_maze.py
│   ├── train_dqn_three_maze.py
│   ├── evaluate_trained_dqn_agent.py
│   ├── evaluate_trained_dqn_medium_maze.py
│   ├── evaluate_trained_dqn_multi_maze.py
│   ├── evaluate_trained_dqn_three_maze.py
│   ├── summarize_final_results.py
│   └── plot_final_comparison.py
│
├── memory/
│   └── replay_buffer.py
│
├── models/
│   └── q_network.py
│
├── results/
│   ├── final_comparison.csv
│   ├── final_plots/
│   └── métricas y gráficas de entrenamiento
│
├── checkpoints/
│   └── modelos entrenados
│
├── requirements.txt
└── README.md
```

---

## 4. Entorno

El entorno representa el laberinto como una matriz bidimensional:

```text
0 = espacio libre
1 = pared
```

El agente se mueve mediante cuatro acciones discretas:

```text
0 = arriba
1 = abajo
2 = izquierda
3 = derecha
```

Cada episodio inicia con el agente en una posición inicial y termina cuando llega a la meta o cuando supera el número máximo de pasos permitido.

Durante el entrenamiento se permitió que las colisiones no terminaran inmediatamente el episodio. Esto ayudó a que el agente pudiera equivocarse, recibir castigo, continuar explorando y aprender de sus errores.

---

## 5. Representación del estado

El estado usado por la red neuronal está formado por 6 valores numéricos:

```text
[
  distancia a pared arriba,
  distancia a pared abajo,
  distancia a pared izquierda,
  distancia a pared derecha,
  delta fila hacia la meta,
  delta columna hacia la meta
]
```

Las primeras cuatro variables funcionan como sensores locales del entorno. Las últimas dos indican la dirección relativa de la meta respecto a la posición actual del agente.

Esta representación permite que el agente perciba paredes cercanas y tenga información sobre la ubicación relativa del objetivo.

---

## 6. Sistema de recompensas

El sistema de recompensas usado fue:

```text
+100  llegar a la meta
-10   colisión
-1    costo base por movimiento
+2    si el agente se acerca a la meta
-2    si el agente se aleja de la meta
-10   penalización por timeout
```

Por lo tanto, un movimiento útil que acerca al agente a la meta tiene recompensa neta:

```text
-1 + 2 = +1
```

Y un movimiento que lo aleja de la meta tiene recompensa neta:

```text
-1 - 2 = -3
```

Este diseño de recompensas busca incentivar trayectorias cortas, evitar paredes y llegar a la meta.

---

## 7. Agentes implementados

### RandomAgent

El `RandomAgent` selecciona acciones aleatorias sin aprender del entorno.

Este agente se usó como línea base para comparar el desempeño de los modelos entrenados.

Su desempeño fue bajo, ya que no tiene memoria, no usa recompensas y no aprende una política.

### DQNAgent

El `DQNAgent` está basado en Deep Q-Learning.

Este agente usa una red neuronal para aproximar valores Q y seleccionar acciones mediante una política epsilon-greedy.

El agente incluye:

- Red neuronal Q.
- Política epsilon-greedy.
- Replay Buffer.
- Entrenamiento por batches.
- Target Network.
- Optimizador Adam.
- Función de pérdida MSE.

---

## 8. Red neuronal Q

La red neuronal recibe el estado del agente y produce cuatro valores Q, uno por cada acción posible.

Arquitectura usada:

```text
Entrada: 6 valores
Capa oculta: 64 neuronas + ReLU
Capa oculta: 64 neuronas + ReLU
Salida: 4 valores Q
```

Cada valor Q representa la estimación del retorno futuro esperado al tomar una acción desde un estado determinado.

Por ejemplo:

```text
Q(s, arriba)
Q(s, abajo)
Q(s, izquierda)
Q(s, derecha)
```

El agente selecciona la acción con mayor valor Q cuando está explotando lo aprendido.

---

## 9. Replay Buffer

El Replay Buffer almacena experiencias del tipo:

```text
estado, acción, recompensa, siguiente estado, done
```

Durante el entrenamiento, el agente toma batches aleatorios de esta memoria.

Esto permite:

- Reutilizar experiencias pasadas.
- Reducir la correlación entre pasos consecutivos.
- Entrenar la red con ejemplos variados.
- Mejorar la estabilidad del aprendizaje.

El Replay Buffer fue una pieza clave para que el agente aprendiera a partir de su interacción con el entorno.

---

## 10. Target Network

Se implementó una Target Network para estabilizar el cálculo de los valores objetivo.

El agente usa dos redes:

```text
q_network        → red principal que se entrena constantemente
target_q_network → red usada para calcular objetivos más estables
```

La red principal se actualiza en cada paso de aprendizaje.

La Target Network se actualiza cada cierto número de episodios copiando los pesos de la red principal:

```text
target_q_network ← q_network
```

Esto evita que el objetivo de aprendizaje cambie demasiado rápido y hace que el entrenamiento sea más estable.

---

## 11. Entrenamiento del agente

El entrenamiento del agente DQN sigue este ciclo:

```text
por cada episodio:
    reiniciar entorno
    mientras el episodio no termine:
        seleccionar acción con epsilon-greedy
        ejecutar acción
        guardar experiencia en Replay Buffer
        tomar batch aleatorio
        calcular Q(s, a)
        calcular target Q
        calcular loss
        actualizar pesos de la red
    reducir epsilon
    actualizar Target Network cada cierto número de episodios
```

La política epsilon-greedy permite equilibrar exploración y explotación:

```text
epsilon alto  → más acciones aleatorias
epsilon bajo  → más acciones elegidas por la red
```

Al inicio del entrenamiento el agente explora mucho. Conforme avanza el entrenamiento, epsilon disminuye y el agente usa más lo aprendido.

---

## 12. Experimentos realizados

Se realizaron los siguientes experimentos:

1. Evaluación del `RandomAgent`.
2. Evaluación de un DQN sin entrenar.
3. Entrenamiento de DQN en `SIMPLE_MAZE`.
4. Entrenamiento de DQN con Target Network.
5. Evaluación del modelo de `SIMPLE_MAZE` en `MEDIUM_MAZE`.
6. Entrenamiento directo en `MEDIUM_MAZE`.
7. Entrenamiento multi-mapa con `SIMPLE_MAZE` y `MEDIUM_MAZE`.
8. Evaluación del modelo multi-mapa en `TEST_MAZE` no visto.
9. Entrenamiento con tres mapas: `SIMPLE_MAZE`, `MEDIUM_MAZE` y `TEST_MAZE`.
10. Comparación final de resultados.
11. Generación de gráficas comparativas finales.
12. Documentación final del proyecto.

---

## 13. Resultados finales

Resumen de resultados principales:

| Modelo | Mapa | Éxito | Reward promedio | Pasos promedio | Colisiones promedio |
|---|---|---:|---:|---:|---:|
| RandomAgent | simple | 0% | -49.92 | 1.84 | 1.00 |
| DQN Simple + Target Network | simple | 100% | 106.00 | 7.00 | 0.00 |
| DQN Medium | medium | 100% | 110.00 | 11.00 | 0.00 |
| DQN Multi-Maze | simple | 100% | 106.00 | 7.00 | 0.00 |
| DQN Multi-Maze | medium | 100% | 110.00 | 11.00 | 0.00 |
| DQN Three-Maze | simple | 100% | 106.00 | 7.00 | 0.00 |
| DQN Three-Maze | medium | 100% | 110.00 | 11.00 | 0.00 |
| DQN Three-Maze | test | 100% | 111.00 | 12.00 | 0.00 |

---

## 14. Interpretación de resultados

El `RandomAgent` obtuvo 0% de éxito en el mapa simple. Esto confirma que el problema no se resuelve eficientemente mediante acciones aleatorias.

El DQN entrenado en `SIMPLE_MAZE` logró resolver el mapa con 100% de éxito, 0 colisiones y una ruta óptima de 7 pasos.

Posteriormente, el agente entrenado directamente en `MEDIUM_MAZE` también aprendió una política óptima, alcanzando la meta en 11 pasos.

El modelo entrenado en `SIMPLE_MAZE` no generalizó directamente a `MEDIUM_MAZE`. Se observó que podía evitar paredes, pero se quedaba atrapado en ciclos sin llegar a la meta.

Después, el modelo entrenado con `SIMPLE_MAZE` y `MEDIUM_MAZE` resolvió ambos mapas, pero no logró resolver correctamente `TEST_MAZE`, un mapa no visto con una distribución distinta.

Finalmente, al entrenar un solo agente con tres mapas distintos, el modelo logró resolver los tres entornos con 100% de éxito, 0 colisiones y rutas óptimas.

---

## 15. Conclusiones

El proyecto demuestra que un agente basado en Deep Q-Learning puede aprender políticas óptimas de navegación en laberintos discretos.

El uso de Replay Buffer permitió entrenar la red a partir de experiencias pasadas, mientras que la Target Network ayudó a estabilizar el cálculo de los valores objetivo.

Los resultados muestran una diferencia clara entre el agente aleatorio y el agente entrenado. Mientras el `RandomAgent` obtuvo 0% de éxito, los modelos DQN entrenados alcanzaron 100% de éxito en los mapas para los que fueron entrenados.

También se observó que entrenar en un solo mapa no produce necesariamente una política general para resolver mapas no vistos. La generalización mejoró al ampliar la distribución de entrenamiento e incluir mapas con trayectorias diferentes.

---

## 16. Limitaciones

El proyecto tiene algunas limitaciones importantes:

- Los mapas usados son pequeños.
- Las posiciones iniciales y metas son fijas.
- El agente no tiene memoria explícita del recorrido.
- El estado contiene información local, pero no un mapa completo.
- La generalización a mapas completamente nuevos sigue siendo limitada.
- Las recompensas fueron diseñadas manualmente.
- No se implementaron variantes avanzadas como Double DQN o Dueling DQN.
- No se usaron mapas generados aleatoriamente.
- El agente no compara su desempeño contra algoritmos clásicos como A*.

Estas limitaciones no invalidan los resultados, pero delimitan el alcance del proyecto.

---

## 17. Trabajo futuro

Algunas mejoras posibles son:

- Entrenar con mapas generados aleatoriamente.
- Variar posiciones iniciales y metas.
- Implementar Double DQN.
- Implementar Dueling DQN.
- Usar Prioritized Experience Replay.
- Comparar contra algoritmos clásicos como A*.
- Entrenar en laberintos más grandes.
- Usar una representación de estado más rica.
- Agregar memoria recurrente para que el agente recuerde posiciones visitadas.
- Evaluar el desempeño en mapas no vistos después de entrenar con una mayor variedad de entornos.

---

## 18. Cómo ejecutar

Instalar dependencias:

```powershell
pip install -r requirements.txt
```

Entrenar el agente en tres mapas:

```powershell
python -m experiments.train_dqn_three_maze
```

Evaluar el agente entrenado en tres mapas:

```powershell
python -m experiments.evaluate_trained_dqn_three_maze
```

Generar comparación final:

```powershell
python -m experiments.summarize_final_results
```

Generar gráficas comparativas finales:

```powershell
python -m experiments.plot_final_comparison
```

Ejecutar demo visual de Pygame para el mapa mediano:

```powershell
python -m experiments.trained_dqn_medium_maze_pygame_demo
```

---

## 19. Estado final

El proyecto finaliza con un agente DQN capaz de resolver tres laberintos vistos durante entrenamiento con:

```text
100% tasa de éxito
0 colisiones
0 timeouts
rutas óptimas
```

El sistema completo incluye:

```text
Entorno funcional
Agente aleatorio
DQN básico
Replay Buffer
Target Network
Entrenamiento por episodios
Evaluación sin exploración
Generalización por múltiples mapas
Checkpoints
CSV de métricas
Gráficas finales
README profesional
```

---

## 20. Gráficas finales

A partir del archivo `results/final_comparison.csv` se generaron gráficas comparativas para visualizar el desempeño final de los agentes y modelos entrenados.

Las gráficas se encuentran en la carpeta:

```text
results/final_plots/
```

Archivos generados:

```text
final_goal_rate.png
final_average_reward.png
final_average_steps.png
final_average_collisions.png
```

Estas gráficas comparan:

- Tasa de éxito por experimento.
- Reward promedio por experimento.
- Pasos promedio por experimento.
- Colisiones promedio por experimento.

Los resultados muestran una mejora clara entre el agente aleatorio y los modelos DQN entrenados. Mientras que el `RandomAgent` obtuvo 0% de éxito, los modelos DQN lograron 100% de éxito en los mapas para los que fueron entrenados.

---

## 21. Resumen final del proyecto

Durante el desarrollo se construyó un sistema completo de aprendizaje por refuerzo para navegación en laberintos.

El avance del proyecto fue progresivo:

```text
RandomAgent
↓
Q-Network
↓
DQNAgent
↓
Replay Buffer
↓
Entrenamiento por batches
↓
Target Network
↓
Entrenamiento por episodios
↓
Evaluación sin exploración
↓
Entrenamiento multi-mapa
↓
Análisis final
```

El resultado final más importante fue el entrenamiento de un solo modelo DQN en tres mapas distintos:

```text
SIMPLE_MAZE
MEDIUM_MAZE
TEST_MAZE
```

La evaluación final del modelo entrenado en tres mapas obtuvo:

| Mapa | Tasa de éxito | Reward promedio | Pasos promedio | Colisiones promedio |
|---|---:|---:|---:|---:|
| SIMPLE_MAZE | 100% | 106.00 | 7.00 | 0.00 |
| MEDIUM_MAZE | 100% | 110.00 | 11.00 | 0.00 |
| TEST_MAZE | 100% | 111.00 | 12.00 | 0.00 |

Esto confirma que el agente aprendió políticas óptimas para los mapas incluidos durante el entrenamiento.

---

## 22. Conclusión final

El proyecto demuestra que un agente DQN puede aprender a navegar en laberintos mediante aprendizaje por refuerzo profundo.

El agente aleatorio no logró resolver el entorno, mientras que los modelos DQN entrenados alcanzaron rutas óptimas, sin colisiones y sin timeouts.

También se observó que entrenar en un solo mapa no garantiza generalización a mapas nuevos. Sin embargo, al ampliar la distribución de entrenamiento e incluir varios mapas, un solo modelo fue capaz de resolver todos los entornos vistos durante el entrenamiento.

Como trabajo futuro, se propone entrenar con mapas generados aleatoriamente, variar posiciones iniciales y metas, implementar Double DQN, Dueling DQN o Prioritized Replay, y comparar el desempeño contra algoritmos clásicos de búsqueda como A*.