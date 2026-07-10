import csv
import os

import torch

from env.maps import TEST_MAZE, TEST_START, TEST_GOAL
from env.maze_env import MazeEnv
from agents.dqn_agent import DQNAgent


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


num_episodes = 20
epsilon = 0.0

results = []

for episode in range(1, num_episodes + 1):
    state = env.reset()
    done = False

    total_reward = 0
    steps = 0
    collision_count = 0
    final_info = None

    while not done:
        action = agent.select_action(state, epsilon=epsilon)

        next_state, reward, done, info = env.step(action)

        if info["collision"]:
            collision_count += 1

        total_reward += reward
        steps += 1
        state = next_state
        final_info = info

    episode_result = {
        "episode": episode,
        "steps": steps,
        "total_reward": total_reward,
        "collision": final_info["collision"],
        "collision_count": collision_count,
        "goal": final_info["goal"],
        "timeout": final_info["timeout"],
    }

    results.append(episode_result)


total_goals = sum(result["goal"] for result in results)
total_timeouts = sum(result["timeout"] for result in results)
total_collision_events = sum(result["collision_count"] for result in results)

average_reward = sum(result["total_reward"] for result in results) / num_episodes
average_steps = sum(result["steps"] for result in results) / num_episodes
average_collision_count = total_collision_events / num_episodes

goal_rate = total_goals / num_episodes * 100
timeout_rate = total_timeouts / num_episodes * 100


print("\nEvaluación del modelo multi-mapa sobre TEST_MAZE no visto")
print(f"Episodios evaluados: {num_episodes}")
print(f"Metas totales: {total_goals}")
print(f"Timeouts totales: {total_timeouts}")
print(f"Colisiones durante episodios: {total_collision_events}")
print(f"Tasa de meta: {goal_rate:.2f}%")
print(f"Tasa de timeout: {timeout_rate:.2f}%")
print(f"Reward promedio: {average_reward:.2f}")
print(f"Pasos promedio: {average_steps:.2f}")
print(f"Colisiones promedio por episodio: {average_collision_count:.2f}")


os.makedirs("results", exist_ok=True)

output_file = "results/dqn_multi_maze_on_test_maze_metrics.csv"

with open(output_file, mode="w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "episode",
            "steps",
            "total_reward",
            "collision",
            "collision_count",
            "goal",
            "timeout",
        ],
    )

    writer.writeheader()
    writer.writerows(results)

print(f"\nMétricas guardadas en: {output_file}")