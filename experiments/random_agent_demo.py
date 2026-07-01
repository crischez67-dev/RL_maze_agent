from env.maps import SIMPLE_MAZE, SIMPLE_START, SIMPLE_GOAL
from env.maze_env import MazeEnv
from agents.random_agent import RandomAgent


env = MazeEnv(
    maze=SIMPLE_MAZE,
    start_pos=SIMPLE_START,
    goal_pos=SIMPLE_GOAL,
    max_steps=50,
)

agent = RandomAgent(num_actions=4)

def run_episode():
    state = env.reset()

    done = False
    step_count = 0
    total_reward = 0
    final_info = None

    while not done:
        action = agent.select_action(state)

        next_state, reward, done, info = env.step(action)

        step_count += 1
        total_reward += reward

        state = next_state
        final_info = info
        final_position = env.agent_pos

    return step_count, total_reward, final_info, final_position

state = env.reset()

print("Estado inicial:", state)
print("Posición inicial:", env.agent_pos)

done = False
step_count = 0
total_reward = 0

num_episodes = 100

total_steps_all = 0
total_reward_all = 0

collisions = 0
goals = 0
timeouts = 0

for episode in range(num_episodes):
    steps, total_reward, info, final_position = run_episode()

    """print("-" * 40)
    print("Episodio:", episode + 1)
    print("Posición final:", final_position)
    print("Pasos:", steps)
    print("Recompensa total:", total_reward)
    print("Terminó por colisión:", info["collision"])
    print("Llegó a la meta:", info["goal"])
    print("Terminó por timeout:", info["timeout"])"""

    total_steps_all += steps
    total_reward_all += total_reward

    if info["collision"]:
        collisions += 1

    if info["goal"]:
        goals += 1

    if info["timeout"]:
        timeouts += 1

print("=" * 40)
print("Resumen general")
print("Episodios totales:", num_episodes)
print("Colisiones:", collisions)
print("Llegadas a la meta:", goals)
print("Timeouts:", timeouts)
print("Promedio de pasos:", total_steps_all / num_episodes)
print("Recompensa promedio:", total_reward_all / num_episodes)