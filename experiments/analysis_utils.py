import pandas as pd
import matplotlib.pyplot as plt

def load_metrics(csv_file):
    df = pd.read_csv(csv_file)
    return df

def compute_summary(df):
    total_episodes = len(df)

    collision_rate = df["collision"].mean() * 100
    goal_rate = df["goal"].mean() * 100
    timeout_rate = df["timeout"].mean() * 100

    average_steps = df["steps"].mean()
    average_reward = df["total_reward"].mean()

    summary = {
        "total_episodes": total_episodes,
        "collision_rate": collision_rate,
        "goal_rate": goal_rate,
        "timeout_rate": timeout_rate,
        "average_steps": average_steps,
        "average_reward": average_reward,
    }

    return summary

def plot_metric(df, x_column, y_column, title, xlabel, ylabel, output_file):
    plt.figure()

    plt.plot(df[x_column], df[y_column])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(True)

    plt.savefig(output_file)
    plt.show()