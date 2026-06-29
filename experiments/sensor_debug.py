from env.maps import SIMPLE_MAZE, SIMPLE_START, SIMPLE_GOAL
from env.maze_env import MazeEnv

def print_state_details(state):
    print("Distancia arriba:", state[0])
    print("Distancia abajo:", state[1])
    print("Distancia izquierda:", state[2])
    print("Distancia derecha:", state[3])
    print("Delta fila hacia meta:", state[4])
    print("Delta columna hacia meta:", state[5])

def run_action(action, action_name):
    print("\n" + "-" * 40)
    print("Ejecutando acción:", action_name)

    next_state, reward, done, info = env.step(action)

    print("Nueva posición:", env.agent_pos)
    print("Recompensa:", reward)
    print("Done:", done)
    print("Info:", info)

    print_state_details(next_state)

    return done

env = MazeEnv(
    maze=SIMPLE_MAZE,
    start_pos=SIMPLE_START,
    goal_pos=SIMPLE_GOAL,
    max_steps=50,
)

print("Posición inicial del agente:", env.agent_pos)
print("Estado inicial:", env.get_state())

state = env.get_state()
print_state_details(state)

# run_action(3, "derecha")
actions = [
    (3, "derecha"),
    (3, "derecha"),
    (3, "derecha"),
    (3, "derecha"),
    (1, "abajo"),
    (1, "abajo"),
    (1, "abajo"),
]

for action, action_name in actions:
    done = run_action(action, action_name)

    if done:
        print("El episodio terminó. Se detiene la secuencia.")
        break