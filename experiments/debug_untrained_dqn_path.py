import torch
import random
import numpy as np

from env.maps import SIMPLE_MAZE, SIMPLE_START, SIMPLE_GOAL
from env.maze_env import MazeEnv
from agents.dqn_agent import DQNAgent

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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
)

agent = DQNAgent(
    state_size=6,
    action_size=4,
    hidden_size=64,
)

state = env.reset()
done = False
step_count = 0
total_reward = 0

print("Inicio:", env.agent_pos)

while not done:
    q_values = agent.get_q_values(state)
    action = agent.select_action(state, epsilon=0.0)

    next_state, reward, done, info = env.step(action)

    step_count += 1
    total_reward += reward

    print("-" * 40)
    print("Paso:", step_count)
    print("Valores Q:", q_values)
    print("Acción:", action, ACTION_NAMES[action])
    print("Posición:", env.agent_pos)
    print("Recompensa:", reward)
    print("Info:", info)

    state = next_state

print("=" * 40)
print("Resumen")
print("Pasos:", step_count)
print("Recompensa total:", total_reward)
print("Posición final:", env.agent_pos)