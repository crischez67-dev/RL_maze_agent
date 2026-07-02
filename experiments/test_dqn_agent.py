from env.maps import SIMPLE_MAZE, SIMPLE_START, SIMPLE_GOAL
from env.maze_env import MazeEnv
from agents.dqn_agent import DQNAgent


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

print("Probando epsilon = 1.0, exploración pura")

for i in range(10):
    action = agent.select_action(state, epsilon=1.0)
    print("Acción exploratoria:", action)

print("\nProbando epsilon = 0.0, explotación pura")

for i in range(10):
    action = agent.select_action(state, epsilon=0.0)
    print("Acción según la red:", action)

print("\nProbando epsilon = 0.5, mezcla exploración/explotación")

for i in range(20):
    action = agent.select_action(state, epsilon=0.5)
    print("Acción con epsilon 0.5:", action)

print("Estado inicial:", state)
print("Posición inicial:", env.agent_pos)

#action = agent.select_action(state, epsilon=1.0)

print("Acción elegida por DQNAgent:", action)

"""next_state, reward, done, info = env.step(action)

print("Nueva posición:", env.agent_pos)
print("Nuevo estado:", next_state)
print("Recompensa:", reward)
print("Done:", done)
print("Info:", info)"""