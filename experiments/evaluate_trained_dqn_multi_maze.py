import csv
import os

import torch

from env.maps import (
    SIMPLE_MAZE,
    SIMPLE_START,
    SIMPLE_GOAL,
    MEDIUM_MAZE,
    MEDIUM_START,
    MEDIUM_GOAL,
)
from env.maze_env import MazeEnv
from agents.dqn_agent import DQNAgent


MAP_CONFIGS = [
    {
        "name": "simple",
        "maze": SIMPLE_MAZE,
        "start": SIMPLE_START,
        "goal": SIMPLE_GOAL,
        "max_steps": 50,
    },
    {
        "name": "medium",
        "maze": MEDIUM_MAZE,
        "start": MEDIUM_START,
        "goal": MEDIUM_GOAL,
        "max_steps": 100,
    },
]


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


num_episodes_per_map = 100
epsilon = 0.0

all_results = []


for map_config in MAP_CONFIGS:
    env = MazeEnv(
        maze=map_config["maze"],
        start_pos=map_config["start"],
        goal_pos=map_config["goal"],
        max_steps=map_config["max_steps"],
        end_on_collision=False,
    )

    results = []

    for episode in range(1, num_episodes_per_map + 1):
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
            "map_name": map_config["name"],
            "episode": episode,
            "steps": steps,
            "total_reward": total_reward,
            "collision": final_info["collision"],
            "collision_count": collision_count,
            "goal": final_info["goal"],
            "timeout": final_info["timeout"],
        }

        results.append(episode_result)
        all_results.append(episode_result)

    total_goals = sum(result["goal"] for result in results)
    total_timeouts = sum(result["timeout"] for result in results)
    total_collision_events = sum(result["collision_count"] for result in results)

    average_reward = sum(result["total_reward"] for result in results) / num_episodes_per_map
    average_steps = sum(result["steps"] for result in results) / num_episodes_per_map
    average_collision_count = total_collision_events / num_episodes_per_map

    goal_rate = total_goals / num_episodes_per_map * 100
    timeout_rate = total_timeouts / num_episodes_per_map * 100

    print(f"\nEvaluación en mapa: {map_config['name']}")
    print(f"Episodios evaluados: {num_episodes_per_map}")
    print(f"Metas totales: {total_goals}")
    print(f"Timeouts totales: {total_timeouts}")
    print(f"Colisiones durante episodios: {total_collision_events}")
    print(f"Tasa de meta: {goal_rate:.2f}%")
    print(f"Tasa de timeout: {timeout_rate:.2f}%")
    print(f"Reward promedio: {average_reward:.2f}")
    print(f"Pasos promedio: {average_steps:.2f}")
    print(f"Colisiones promedio por episodio: {average_collision_count:.2f}")


os.makedirs("results", exist_ok=True)

output_file = "results/trained_dqn_multi_maze_metrics.csv"

with open(output_file, mode="w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "map_name",
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
    writer.writerows(all_results)

print(f"\nMétricas guardadas en: {output_file}")