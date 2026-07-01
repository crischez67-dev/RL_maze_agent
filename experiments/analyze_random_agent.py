import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "results/random_agent_metrics.csv"

df = pd.read_csv(CSV_FILE)

print("Primeras filas:")
print(df.head())

print("\nInformación general:")
print(df.describe())

total_episodes = len(df)

collision_rate = df["collision"].mean() * 100
goal_rate = df["goal"].mean() * 100
timeout_rate = df["timeout"].mean() * 100

average_steps = df["steps"].mean()
average_reward = df["total_reward"].mean()

print("\nMétricas del agente aleatorio:")
print("Episodios totales:", total_episodes)
print("Tasa de colisión:", collision_rate, "%")
print("Tasa de éxito:", goal_rate, "%")
print("Tasa de timeout:", timeout_rate, "%")
print("Promedio de pasos:", average_steps)
print("Recompensa promedio:", average_reward)

plt.figure()

plt.plot(df["episode"], df["total_reward"])

plt.title("Recompensa total por episodio - RandomAgent")
plt.xlabel("Episodio")
plt.ylabel("Recompensa total")

plt.grid(True)
plt.savefig("results/random_agent_rewards.png")
plt.show()

plt.figure()

plt.plot(df["episode"], df["steps"])

plt.title("Pasos por episodio - RandomAgent")
plt.xlabel("Episodio")
plt.ylabel("Pasos")

plt.grid(True)

plt.savefig("results/random_agent_steps.png")
plt.show()