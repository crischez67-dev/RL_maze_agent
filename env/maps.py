# env/maps.py

"""
Mapas iniciales para el proyecto RL Maze Agent.

Convención:
0 = espacio libre
1 = pared

Las coordenadas se manejan como:
(fila, columna)
"""

SIMPLE_MAZE = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

SIMPLE_START = (1, 1)
SIMPLE_GOAL = (4, 5)