import os
import pandas as pd


EXPERIMENTS = [
    {
        "model": "RandomAgent",
        "map": "simple",
        "file": "results/random_agent_metrics.csv",
        "map_filter": None,
    },
    {
        "model": "DQN Simple + Target Network",
        "map": "simple",
        "file": "results/trained_dqn_agent_target_network_metrics.csv",
        "map_filter": None,
    },
    {
        "model": "DQN Medium",
        "map": "medium",
        "file": "results/trained_dqn_medium_maze_metrics.csv",
        "map_filter": None,
    },
    {
        "model": "DQN Multi-Maze",
        "map": "simple",
        "file": "results/trained_dqn_multi_maze_metrics.csv",
        "map_filter": "simple",
    },
    {
        "model": "DQN Multi-Maze",
        "map": "medium",
        "file": "results/trained_dqn_multi_maze_metrics.csv",
        "map_filter": "medium",
    },
    {
        "model": "DQN Three-Maze",
        "map": "simple",
        "file": "results/trained_dqn_three_maze_metrics.csv",
        "map_filter": "simple",
    },
    {
        "model": "DQN Three-Maze",
        "map": "medium",
        "file": "results/trained_dqn_three_maze_metrics.csv",
        "map_filter": "medium",
    },
    {
        "model": "DQN Three-Maze",
        "map": "test",
        "file": "results/trained_dqn_three_maze_metrics.csv",
        "map_filter": "test",
    },
]


def summarize_experiment(experiment):
    file_path = experiment["file"]

    if not os.path.exists(file_path):
        print(f"Archivo no encontrado: {file_path}")
        return None

    df = pd.read_csv(file_path)

    if experiment["map_filter"] is not None:
        df = df[df["map_name"] == experiment["map_filter"]]

    total_episodes = len(df)

    if total_episodes == 0:
        return None

    goal_rate = df["goal"].mean() * 100 if "goal" in df.columns else None
    timeout_rate = df["timeout"].mean() * 100 if "timeout" in df.columns else None
    average_reward = df["total_reward"].mean() if "total_reward" in df.columns else None
    average_steps = df["steps"].mean() if "steps" in df.columns else None

    if "collision_count" in df.columns:
        average_collisions = df["collision_count"].mean()
    elif "collision" in df.columns:
        average_collisions = df["collision"].mean()
    else:
        average_collisions = None

    return {
        "model": experiment["model"],
        "map": experiment["map"],
        "episodes": total_episodes,
        "goal_rate_percent": goal_rate,
        "timeout_rate_percent": timeout_rate,
        "average_reward": average_reward,
        "average_steps": average_steps,
        "average_collisions": average_collisions,
        "source_file": file_path,
    }


summaries = []

for experiment in EXPERIMENTS:
    summary = summarize_experiment(experiment)

    if summary is not None:
        summaries.append(summary)


summary_df = pd.DataFrame(summaries)

os.makedirs("results", exist_ok=True)

output_file = "results/final_comparison.csv"
summary_df.to_csv(output_file, index=False)

print("\nComparación final:")
print(summary_df)

print(f"\nTabla comparativa guardada en: {output_file}")