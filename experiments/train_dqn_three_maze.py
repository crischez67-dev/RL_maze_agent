import csv
import os
import random

import numpy as np
import torch

from env.maps import (
    SIMPLE_MAZE,
    SIMPLE_START,
    SIMPLE_GOAL,
    MEDIUM_MAZE,
    MEDIUM_START,
    MEDIUM_GOAL,
    TEST_MAZE,
    TEST_START,
    TEST_GOAL,
)
from env.maze_env import MazeEnv
from agents.dqn_agent import DQNAgent
from memory.replay_buffer import ReplayBuffer


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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
    {
        "name": "test",
        "maze": TEST_MAZE,
        "start": TEST_START,
        "goal": TEST_GOAL,
        "max_steps": 120,
    },
]


agent = DQNAgent(
    state_size=6,
    action_size=4,
    hidden_size=64,
    learning_rate=0.001,
)

memory = ReplayBuffer(capacity=50000)


num_episodes = 2000
batch_size = 64
gamma = 0.99

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

target_update_frequency = 10

results = []


for episode in range(1, num_episodes + 1):
    map_config = random.choice(MAP_CONFIGS)

    env = MazeEnv(
        maze=map_config["maze"],
        start_pos=map_config["start"],
        goal_pos=map_config["goal"],
        max_steps=map_config["max_steps"],
        end_on_collision=False,
    )

    state = env.reset()
    done = False

    total_reward = 0
    losses = []
    final_info = None
    steps = 0
    collision_count = 0

    while not done:
        action = agent.select_action(state, epsilon=epsilon)

        next_state, reward, done, info = env.step(action)

        if info["collision"]:
            collision_count += 1

        memory.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        loss = agent.learn(
            memory=memory,
            batch_size=batch_size,
            gamma=gamma,
        )

        if loss is not None:
            losses.append(loss)

        total_reward += reward
        steps += 1
        state = next_state
        final_info = info

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % target_update_frequency == 0:
        agent.update_target_network()

    average_loss = sum(losses) / len(losses) if losses else None

    episode_result = {
        "episode": episode,
        "map_name": map_config["name"],
        "steps": steps,
        "total_reward": total_reward,
        "collision": final_info["collision"],
        "collision_count": collision_count,
        "goal": final_info["goal"],
        "timeout": final_info["timeout"],
        "epsilon": epsilon,
        "average_loss": average_loss,
    }

    results.append(episode_result)

    if episode % 50 == 0:
        recent_results = results[-50:]

        recent_goals = sum(result["goal"] for result in recent_results)
        recent_timeouts = sum(result["timeout"] for result in recent_results)
        recent_collision_count = sum(result["collision_count"] for result in recent_results)
        recent_average_reward = sum(result["total_reward"] for result in recent_results) / len(recent_results)
        recent_average_steps = sum(result["steps"] for result in recent_results) / len(recent_results)

        simple_results = [result for result in recent_results if result["map_name"] == "simple"]
        medium_results = [result for result in recent_results if result["map_name"] == "medium"]
        test_results = [result for result in recent_results if result["map_name"] == "test"]

        simple_goals = sum(result["goal"] for result in simple_results)
        medium_goals = sum(result["goal"] for result in medium_results)
        test_goals = sum(result["goal"] for result in test_results)

        print(
            f"Episodio {episode:04d} | "
            f"epsilon={epsilon:.3f} | "
            f"reward_prom_50={recent_average_reward:.2f} | "
            f"metas_50={recent_goals} | "
            f"timeouts_50={recent_timeouts} | "
            f"colisiones_prom_50={recent_collision_count / 50:.2f} | "
            f"pasos_prom_50={recent_average_steps:.2f} | "
            f"simple={simple_goals}/{len(simple_results)} | "
            f"medium={medium_goals}/{len(medium_results)} | "
            f"test={test_goals}/{len(test_results)}"
        )


os.makedirs("results", exist_ok=True)

output_file = "results/dqn_three_maze_training_metrics.csv"

with open(output_file, mode="w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "episode",
            "map_name",
            "steps",
            "total_reward",
            "collision",
            "collision_count",
            "goal",
            "timeout",
            "epsilon",
            "average_loss",
        ],
    )

    writer.writeheader()
    writer.writerows(results)


os.makedirs("checkpoints", exist_ok=True)

model_file = "checkpoints/dqn_three_maze.pth"
torch.save(agent.q_network.state_dict(), model_file)

print("\nEntrenamiento con tres mapas terminado.")
print(f"Métricas guardadas en: {output_file}")
print(f"Modelo guardado en: {model_file}")