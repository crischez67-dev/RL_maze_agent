import pandas as pd
from experiments.analysis_utils import load_metrics, compute_summary, plot_metric

CSV_FILE = "results/random_agent_metrics.csv"

df = load_metrics(CSV_FILE)

print("Primeras filas:")
print(df.head())

print("\nInformación general:")
print(df.describe())

summary = compute_summary(df)

print("\nMétricas del agente aleatorio:")
print("Episodios totales:", summary["total_episodes"])
print("Tasa de colisión:", summary["collision_rate"], "%")
print("Tasa de éxito:", summary["goal_rate"], "%")
print("Tasa de timeout:", summary["timeout_rate"], "%")
print("Promedio de pasos:", summary["average_steps"])
print("Recompensa promedio:", summary["average_reward"])

plot_metric(
    df=df,
    x_column="episode",
    y_column="total_reward",
    title="Recompensa total por episodio - RandomAgent",
    xlabel="Episodio",
    ylabel="Recompensa total",
    output_file="results/random_agent_rewards.png",
)

plot_metric(
    df=df,
    x_column="episode",
    y_column="steps",
    title="Pasos por episodio - RandomAgent",
    xlabel="Episodio",
    ylabel="Pasos",
    output_file="results/random_agent_steps.png",
)