# env/renderer.py

import pygame


class MazeRenderer:
    """
    Renderizador visual del laberinto usando Pygame.

    Esta clase no contiene lógica de aprendizaje por refuerzo.
    Solo dibuja el estado actual del entorno.
    """

    def __init__(self, env, cell_size=80):
        self.env = env
        self.cell_size = cell_size

        self.width = self.env.cols * self.cell_size
        self.height = self.env.rows * self.cell_size

        pygame.init()

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RL Maze Agent - Demo Manual")

        self.clock = pygame.time.Clock()

        self.wall_color = (40, 40, 40)
        self.free_color = (235, 235, 235)
        self.goal_color = (80, 200, 120)
        self.agent_color = (70, 130, 255)
        self.grid_color = (180, 180, 180)

    def draw(self):
        """
        Dibuja el laberinto, la meta y el agente.
        """
        self.screen.fill((255, 255, 255))

        for row in range(self.env.rows):
            for col in range(self.env.cols):
                self.draw_cell(row, col)

        self.draw_agent()

        pygame.display.flip()

    def draw_cell(self, row, col):
        """
        Dibuja una celda individual del mapa.
        """
        rect = pygame.Rect(
            col * self.cell_size,
            row * self.cell_size,
            self.cell_size,
            self.cell_size,
        )

        pos = (row, col)

        if self.env.maze[row][col] == 1:
            color = self.wall_color
        elif pos == self.env.goal_pos:
            color = self.goal_color
        else:
            color = self.free_color

        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, self.grid_color, rect, 1)

    def draw_agent(self):
        """
        Dibuja al agente como un círculo.
        """
        row, col = self.env.agent_pos

        center_x = col * self.cell_size + self.cell_size // 2
        center_y = row * self.cell_size + self.cell_size // 2

        radius = self.cell_size // 3

        pygame.draw.circle(
            self.screen,
            self.agent_color,
            (center_x, center_y),
            radius,
        )

    def tick(self, fps=30):
        """
        Controla los FPS de la ventana.
        """
        self.clock.tick(fps)

    def close(self):
        """
        Cierra Pygame.
        """
        pygame.quit()