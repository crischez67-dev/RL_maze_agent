# env/maze_env.py

import numpy as np


class MazeEnv:
    """
    Entorno básico de laberinto para aprendizaje por refuerzo.

    El agente se mueve en una matriz 2D.

    Convención:
    0 = celda libre
    1 = pared

    Acciones:
    0 = arriba
    1 = abajo
    2 = izquierda
    3 = derecha
    """

    ACTIONS = {
        0: (-1, 0),  # arriba
        1: (1, 0),   # abajo
        2: (0, -1),  # izquierda
        3: (0, 1),   # derecha
    }

    def __init__(self, maze, start_pos, goal_pos, max_steps=100):
        self.maze = maze
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.max_steps = max_steps

        self.rows = len(maze)
        self.cols = len(maze[0])

        self.agent_pos = None
        self.steps = 0

        self.reset()

    def reset(self):
        """
        Reinicia el episodio.
        """
        self.agent_pos = self.start_pos
        self.steps = 0
        return self.get_state()

    def is_wall(self, pos):
        """
        Revisa si una posición es pared o está fuera del mapa.
        """
        row, col = pos

        if row < 0 or row >= self.rows:
            return True

        if col < 0 or col >= self.cols:
            return True

        return self.maze[row][col] == 1

    def is_goal(self):
        """
        Revisa si el agente llegó a la meta.
        """
        return self.agent_pos == self.goal_pos

    def distance_to_goal(self, pos=None):
        """
        Calcula la distancia Manhattan entre el agente y la meta.
        """
        if pos is None:
            pos = self.agent_pos

        row, col = pos
        goal_row, goal_col = self.goal_pos

        return abs(goal_row - row) + abs(goal_col - col)

    def get_sensor_distance(self, direction, max_distance=5):
        """
        Sensor simple que mide cuántas celdas libres hay
        en una dirección antes de encontrar una pared.

        direction:
        (-1, 0) = arriba
        (1, 0) = abajo
        (0, -1) = izquierda
        (0, 1) = derecha
        """
        row, col = self.agent_pos
        d_row, d_col = direction

        distance = 0

        for i in range(1, max_distance + 1):
            check_pos = (row + d_row * i, col + d_col * i)

            if self.is_wall(check_pos):
                break

            distance += 1

        return distance / max_distance

    def get_state(self):
        """
        Estado numérico del agente.

        Incluye:
        - distancia a pared arriba
        - distancia a pared abajo
        - distancia a pared izquierda
        - distancia a pared derecha
        - diferencia normalizada en filas hacia la meta
        - diferencia normalizada en columnas hacia la meta
        """
        dist_up = self.get_sensor_distance((-1, 0))
        dist_down = self.get_sensor_distance((1, 0))
        dist_left = self.get_sensor_distance((0, -1))
        dist_right = self.get_sensor_distance((0, 1))

        agent_row, agent_col = self.agent_pos
        goal_row, goal_col = self.goal_pos

        delta_row = (goal_row - agent_row) / self.rows
        delta_col = (goal_col - agent_col) / self.cols

        state = np.array(
            [
                dist_up,
                dist_down,
                dist_left,
                dist_right,
                delta_row,
                delta_col,
            ],
            dtype=np.float32,
        )

        return state

    def step(self, action):
        """
        Ejecuta una acción.

        Retorna:
        next_state, reward, done, info
        """
        if action not in self.ACTIONS:
            raise ValueError(f"Acción inválida: {action}")

        self.steps += 1

        old_pos = self.agent_pos
        old_distance = self.distance_to_goal(old_pos)

        move_row, move_col = self.ACTIONS[action]
        new_pos = (old_pos[0] + move_row, old_pos[1] + move_col)

        info = {
            "collision": False,
            "goal": False,
            "timeout": False,
        }

        # Caso 1: choque contra pared
        if self.is_wall(new_pos):
            reward = -50
            done = True
            info["collision"] = True
            return self.get_state(), reward, done, info

        # Movimiento válido
        self.agent_pos = new_pos
        new_distance = self.distance_to_goal(new_pos)

        # Recompensa base por cada paso
        reward = -1

        # Caso 2: llegó a la meta
        if self.is_goal():
            reward = 100
            done = True
            info["goal"] = True
            return self.get_state(), reward, done, info

        # Caso 3: se acerca o se aleja de la meta
        if new_distance < old_distance:
            reward += 2
        else:
            reward -= 2

        # Caso 4: excedió máximo de pasos
        if self.steps >= self.max_steps:
            reward -= 10
            done = True
            info["timeout"] = True
            return self.get_state(), reward, done, info

        done = False
        return self.get_state(), reward, done, info

    def render_console(self):
        """
        Muestra el laberinto en consola.
        """
        for row in range(self.rows):
            line = ""

            for col in range(self.cols):
                pos = (row, col)

                if pos == self.agent_pos:
                    line += "A "
                elif pos == self.goal_pos:
                    line += "G "
                elif self.maze[row][col] == 1:
                    line += "# "
                else:
                    line += ". "

            print(line)

        print()