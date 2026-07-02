import os

from env.maps import SIMPLE_MAZE, SIMPLE_START, SIMPLE_GOAL
from env.maze_env import MazeEnv
from agents.dqn_agent import DQNAgent
from experiments.evaluation_utils import evaluate_agent


RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "untrained_dqn_agent_metrics.csv")


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

summary = evaluate_agent(
    env=env,
    agent=agent,
    output_file=OUTPUT_FILE,
    num_episodes=100,
)

print("Evaluación terminada.")
print("Archivo guardado en:", OUTPUT_FILE)
print("=" * 40)
print("Resumen general")
print("Episodios totales:", summary["episodes"])
print("Colisiones:", summary["collisions"])
print("Llegadas a la meta:", summary["goals"])
print("Timeouts:", summary["timeouts"])
print("Promedio de pasos:", summary["average_steps"])
print("Recompensa promedio:", summary["average_reward"])