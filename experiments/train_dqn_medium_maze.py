import csv
import os
import random

import numpy as np
import torch

from env.maps import MEDIUM_MAZE, MEDIUM_START, MEDIUM_GOAL
from env.maze_env import MazeEnv
from agents.dqn_agent import DQNAgent
from memory.replay_buffer import ReplayBuffer


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


env = MazeEnv(
    maze=MEDIUM_MAZE,
    start_pos=MEDIUM_START,
    goal_pos=MEDIUM_GOAL,
    max_steps=100,
    end_on_collision=False,
)

agent = DQNAgent(
    state_size=6,
    action_size=4,
    hidden_size=64,
    learning_rate=0.001,
)

memory = ReplayBuffer(capacity=20000)


num_episodes = 1000
batch_size = 64
gamma = 0.99

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

target_update_frequency = 10

results = []


for episode in range(1, num_episodes + 1):
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

        print(
            f"Episodio {episode:04d} | "
            f"epsilon={epsilon:.3f} | "
            f"reward_prom_50={recent_average_reward:.2f} | "
            f"metas_50={recent_goals} | "
            f"timeouts_50={recent_timeouts} | "
            f"colisiones_prom_50={recent_collision_count / 50:.2f} | "
            f"pasos_prom_50={recent_average_steps:.2f}"
        )


os.makedirs("results", exist_ok=True)

output_file = "results/dqn_medium_maze_training_metrics.csv"

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
            "epsilon",
            "average_loss",
        ],
    )

    writer.writeheader()
    writer.writerows(results)


os.makedirs("checkpoints", exist_ok=True)

model_file = "checkpoints/dqn_medium_maze.pth"
torch.save(agent.q_network.state_dict(), model_file)

print("\nEntrenamiento terminado.")
print(f"Métricas guardadas en: {output_file}")
print(f"Modelo guardado en: {model_file}")