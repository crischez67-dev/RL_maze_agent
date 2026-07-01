import csv
import os

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

RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "random_agent_metrics.csv")


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


def evaluate_agent(num_episodes=100):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    total_steps_all = 0
    total_reward_all = 0

    collisions = 0
    goals = 0
    timeouts = 0

    with open(OUTPUT_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "episode",
            "steps",
            "total_reward",
            "collision",
            "goal",
            "timeout",
            "final_row",
            "final_col",
        ])

        for episode in range(1, num_episodes + 1):
            steps, total_reward, info, final_position = run_episode()

            final_row, final_col = final_position

            writer.writerow([
                episode,
                steps,
                total_reward,
                info["collision"],
                info["goal"],
                info["timeout"],
                final_row,
                final_col,
            ])

            total_steps_all += steps
            total_reward_all += total_reward

            if info["collision"]:
                collisions += 1

            if info["goal"]:
                goals += 1

            if info["timeout"]:
                timeouts += 1

    print("Evaluación terminada.")
    print("Archivo guardado en:", OUTPUT_FILE)
    print("=" * 40)
    print("Resumen general")
    print("Episodios totales:", num_episodes)
    print("Colisiones:", collisions)
    print("Llegadas a la meta:", goals)
    print("Timeouts:", timeouts)
    print("Promedio de pasos:", total_steps_all / num_episodes)
    print("Recompensa promedio:", total_reward_all / num_episodes)


evaluate_agent(num_episodes=100)