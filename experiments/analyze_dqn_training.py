import os

import pandas as pd
import matplotlib.pyplot as plt


CSV_FILE = "results/dqn_training_metrics.csv"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE)

print("\nPrimeras filas:")
print(df.head())

print("\nÚltimas filas:")
print(df.tail())

print("\nResumen general:")
print(df.describe())

total_episodes = len(df)
total_goals = df["goal"].sum()
total_collisions = df["collision"].sum()
total_timeouts = df["timeout"].sum()

goal_rate = df["goal"].mean() * 100
collision_rate = df["collision"].mean() * 100
timeout_rate = df["timeout"].mean() * 100
average_reward = df["total_reward"].mean()
average_steps = df["steps"].mean()
total_collision_events = df["collision_count"].sum()
average_collision_count = df["collision_count"].mean()

print("\nMétricas globales:")
print(f"Episodios totales: {total_episodes}")
print(f"Metas totales: {total_goals}")
print(f"Colisiones totales: {total_collisions}")
print(f"Timeouts totales: {total_timeouts}")
print(f"Tasa de meta: {goal_rate:.2f}%")
print(f"Tasa de colisión: {collision_rate:.2f}%")
print(f"Tasa de timeout: {timeout_rate:.2f}%")
print(f"Reward promedio: {average_reward:.2f}")
print(f"Pasos promedio: {average_steps:.2f}")
print(f"Colisiones totales durante episodios: {total_collision_events}")
print(f"Colisiones promedio por episodio: {average_collision_count:.2f}")


window = 20

df["rolling_reward"] = df["total_reward"].rolling(window).mean()
df["rolling_goal_rate"] = df["goal"].rolling(window).mean() * 100
df["rolling_collision_rate"] = df["collision"].rolling(window).mean() * 100
df["rolling_loss"] = df["average_loss"].rolling(window).mean()
df["rolling_collision_count"] = df["collision_count"].rolling(window).mean()


plt.figure()
plt.plot(df["episode"], df["total_reward"])
plt.title("Recompensa total por episodio")
plt.xlabel("Episodio")
plt.ylabel("Recompensa total")
plt.grid(True)
plt.savefig("results/dqn_total_reward.png")
plt.show()


plt.figure()
plt.plot(df["episode"], df["rolling_reward"])
plt.title(f"Recompensa promedio móvil ({window} episodios)")
plt.xlabel("Episodio")
plt.ylabel("Recompensa promedio")
plt.grid(True)
plt.savefig("results/dqn_rolling_reward.png")
plt.show()


plt.figure()
plt.plot(df["episode"], df["rolling_goal_rate"])
plt.title(f"Tasa de meta móvil ({window} episodios)")
plt.xlabel("Episodio")
plt.ylabel("Tasa de meta (%)")
plt.grid(True)
plt.savefig("results/dqn_rolling_goal_rate.png")
plt.show()


plt.figure()
plt.plot(df["episode"], df["rolling_collision_rate"])
plt.title(f"Tasa de colisión móvil ({window} episodios)")
plt.xlabel("Episodio")
plt.ylabel("Tasa de colisión (%)")
plt.grid(True)
plt.savefig("results/dqn_rolling_collision_rate.png")
plt.show()


plt.figure()
plt.plot(df["episode"], df["epsilon"])
plt.title("Epsilon durante el entrenamiento")
plt.xlabel("Episodio")
plt.ylabel("Epsilon")
plt.grid(True)
plt.savefig("results/dqn_epsilon.png")
plt.show()


plt.figure()
plt.plot(df["episode"], df["rolling_loss"])
plt.title(f"Loss promedio móvil ({window} episodios)")
plt.xlabel("Episodio")
plt.ylabel("Loss promedio")
plt.grid(True)
plt.savefig("results/dqn_rolling_loss.png")
plt.show()

plt.figure()
plt.plot(df["episode"], df["rolling_collision_count"])
plt.title(f"Colisiones promedio por episodio ({window} episodios)")
plt.xlabel("Episodio")
plt.ylabel("Colisiones promedio")
plt.grid(True)
plt.savefig("results/dqn_rolling_collision_count.png")
plt.show()