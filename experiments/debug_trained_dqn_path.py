import torch

from env.maps import SIMPLE_MAZE, SIMPLE_START, SIMPLE_GOAL
from env.maze_env import MazeEnv
from agents.dqn_agent import DQNAgent


ACTION_NAMES = {
    0: "arriba",
    1: "abajo",
    2: "izquierda",
    3: "derecha",
}


env = MazeEnv(
    maze=SIMPLE_MAZE,
    start_pos=SIMPLE_START,
    goal_pos=SIMPLE_GOAL,
    max_steps=50,
    end_on_collision=False,
)

agent = DQNAgent(
    state_size=6,
    action_size=4,
    hidden_size=64,
    learning_rate=0.001,
)

model_file = "checkpoints/dqn_agent.pth"

agent.q_network.load_state_dict(
    torch.load(model_file, map_location="cpu")
)

agent.q_network.eval()


state = env.reset()
done = False

total_reward = 0
step_number = 0

print("Ruta aprendida por el agente entrenado")
print("Posición inicial:", env.agent_pos)
print("Meta:", env.goal_pos)

while not done:
    q_values = agent.get_q_values(state)
    action = agent.select_action(state, epsilon=0.0)

    next_state, reward, done, info = env.step(action)

    step_number += 1
    total_reward += reward

    print("\nPaso:", step_number)
    print("Q-values:", q_values)
    print("Acción elegida:", action, "-", ACTION_NAMES[action])
    print("Nueva posición:", env.agent_pos)
    print("Reward:", reward)
    print("Info:", info)

    state = next_state

print("\nEpisodio terminado")
print("Total reward:", total_reward)
print("Pasos:", step_number)