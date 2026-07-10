import torch

from env.maps import TEST_MAZE, TEST_START, TEST_GOAL
from env.maze_env import MazeEnv
from agents.dqn_agent import DQNAgent


ACTION_NAMES = {
    0: "arriba",
    1: "abajo",
    2: "izquierda",
    3: "derecha",
}


env = MazeEnv(
    maze=TEST_MAZE,
    start_pos=TEST_START,
    goal_pos=TEST_GOAL,
    max_steps=120,
    end_on_collision=False,
)

agent = DQNAgent(
    state_size=6,
    action_size=4,
    hidden_size=64,
    learning_rate=0.001,
)

model_file = "checkpoints/dqn_multi_maze.pth"

agent.q_network.load_state_dict(
    torch.load(model_file, map_location="cpu")
)

agent.q_network.eval()


state = env.reset()
done = False

total_reward = 0
step_number = 0
collision_count = 0

visited_positions = {}

print("Ruta del modelo multi-mapa evaluado en TEST_MAZE")
print("Posición inicial:", env.agent_pos)
print("Meta:", env.goal_pos)

while not done:
    position_before = env.agent_pos

    q_values = agent.get_q_values(state)
    action = agent.select_action(state, epsilon=0.0)

    next_state, reward, done, info = env.step(action)

    if info["collision"]:
        collision_count += 1

    step_number += 1
    total_reward += reward

    position_after = env.agent_pos
    visited_positions[position_after] = visited_positions.get(position_after, 0) + 1

    if step_number <= 30:
        print("\nPaso:", step_number)
        print("Posición antes:", position_before)
        print("Q-values:", q_values)
        print("Acción elegida:", action, "-", ACTION_NAMES[action])
        print("Posición después:", position_after)
        print("Reward:", reward)
        print("Info:", info)

    state = next_state

print("\nEpisodio terminado")
print("Total reward:", total_reward)
print("Pasos:", step_number)
print("Colisiones:", collision_count)
print("Info final:", info)

print("\nPosiciones más repetidas:")
for position, count in sorted(visited_positions.items(), key=lambda item: item[1], reverse=True)[:10]:
    print(position, "→", count, "veces")