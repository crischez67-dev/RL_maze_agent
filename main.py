# main.py

from env.maps import SIMPLE_MAZE, SIMPLE_START, SIMPLE_GOAL
from env.maze_env import MazeEnv


def main():
    env = MazeEnv(
        maze=SIMPLE_MAZE,
        start_pos=SIMPLE_START,
        goal_pos=SIMPLE_GOAL,
        max_steps=50,
    )

    done = False

    controls = {
        "w": 0,  # arriba
        "s": 1,  # abajo
        "a": 2,  # izquierda
        "d": 3,  # derecha
    }

    print("Proyecto RL Maze Agent")
    print("Controles:")
    print("w = arriba")
    print("s = abajo")
    print("a = izquierda")
    print("d = derecha")
    print("q = salir")
    print()

    while not done:
        env.render_console()
        print("Estado actual:", env.get_state())
        print("Pasos:", env.steps)

        key = input("Movimiento: ").lower().strip()

        if key == "q":
            print("Ejecución terminada por el usuario.")
            break

        if key not in controls:
            print("Tecla inválida. Usa w, a, s, d o q.")
            continue

        action = controls[key]
        next_state, reward, done, info = env.step(action)

        print("Recompensa:", reward)
        print("Info:", info)
        print("-" * 40)

        if info["collision"]:
            print("El agente chocó contra una pared.")

        if info["goal"]:
            print("El agente llegó a la meta.")

        if info["timeout"]:
            print("Se alcanzó el límite máximo de pasos.")

    print()
    print("Estado final:")
    env.render_console()


if __name__ == "__main__":
    main()