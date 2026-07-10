import os

import pandas as pd
import matplotlib.pyplot as plt


CSV_FILE = "results/final_comparison.csv"
OUTPUT_DIR = "results/final_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE)

df["experiment"] = df["model"] + " - " + df["map"]


def save_bar_plot(column, title, ylabel, output_filename):
    plt.figure(figsize=(12, 6))
    plt.bar(df["experiment"], df[column])
    plt.title(title)
    plt.xlabel("Experimento")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(output_path)
    plt.show()

    print(f"Gráfica guardada en: {output_path}")


save_bar_plot(
    column="goal_rate_percent",
    title="Tasa de éxito por experimento",
    ylabel="Tasa de éxito (%)",
    output_filename="final_goal_rate.png",
)

save_bar_plot(
    column="average_reward",
    title="Reward promedio por experimento",
    ylabel="Reward promedio",
    output_filename="final_average_reward.png",
)

save_bar_plot(
    column="average_steps",
    title="Pasos promedio por experimento",
    ylabel="Pasos promedio",
    output_filename="final_average_steps.png",
)

save_bar_plot(
    column="average_collisions",
    title="Colisiones promedio por experimento",
    ylabel="Colisiones promedio",
    output_filename="final_average_collisions.png",
)

print("\nGráficas finales generadas correctamente.")