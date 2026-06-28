# experiments/demo_pygame.py

import pygame

from env.maps import SIMPLE_MAZE, SIMPLE_START, SIMPLE_GOAL
from env.maze_env import MazeEnv
from env.renderer import MazeRenderer


def main():
    env = MazeEnv(
        maze=SIMPLE_MAZE,
        start_pos=SIMPLE_START,
        goal_pos=SIMPLE_GOAL,
        max_steps=50,
    )

    renderer = MazeRenderer(env, cell_size=80)

    running = True
    done = False

    controls = {
        pygame.K_w: 0,
        pygame.K_s: 1,
        pygame.K_a: 2,
        pygame.K_d: 3,
        pygame.K_UP: 0,
        pygame.K_DOWN: 1,
        pygame.K_LEFT: 2,
        pygame.K_RIGHT: 3,
    }

    print("Demo visual iniciada.")
    print("Controles:")
    print("W/A/S/D o flechas = mover agente")
    print("R = reiniciar episodio")
    print("ESC = salir")

    while running:
        renderer.draw()
        renderer.tick(fps=30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    env.reset()
                    done = False
                    print("Episodio reiniciado.")

                elif event.key in controls and not done:
                    action = controls[event.key]

                    next_state, reward, done, info = env.step(action)

                    print("-" * 40)
                    print("Acción:", action)
                    print("Estado:", next_state)
                    print("Recompensa:", reward)
                    print("Info:", info)

                    if info["collision"]:
                        print("Colisión detectada. Presiona R para reiniciar.")

                    if info["goal"]:
                        print("Meta alcanzada. Presiona R para reiniciar.")

                    if info["timeout"]:
                        print("Límite de pasos alcanzado. Presiona R para reiniciar.")

    renderer.close()


if __name__ == "__main__":
    main()