import pygame
import torch

from env.maps import SIMPLE_MAZE, SIMPLE_START, SIMPLE_GOAL
from env.maze_env import MazeEnv
from env.renderer import MazeRenderer
from agents.dqn_agent import DQNAgent


ACTION_NAMES = {
    0: "arriba",
    1: "abajo",
    2: "izquierda",
    3: "derecha",
}


env = MazeEnv(
    maze=SIMPLE_MAZE,
    start_pos=SIMPLE_START,
    goal_pos=SIMPLE_GOAL,
    max_steps=50,
    end_on_collision=False,
)

agent = DQNAgent(
    state_size=6,
    action_size=4,
    hidden_size=64,
    learning_rate=0.001,
)

model_file = "checkpoints/dqn_agent.pth"

agent.q_network.load_state_dict(
    torch.load(model_file, map_location="cpu")
)

agent.q_network.eval()

renderer = MazeRenderer(env)

state = env.reset()
done = False

total_reward = 0
step_count = 0
last_action = None
last_reward = None
last_info = None

action_delay_ms = 500
last_action_time = pygame.time.get_ticks()

running = True

print("Demo visual del agente DQN entrenado")
print("R = reiniciar")
print("ESC = salir")

while running:
    current_time = pygame.time.get_ticks()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            if event.key == pygame.K_r:
                state = env.reset()
                done = False
                total_reward = 0
                step_count = 0
                last_action = None
                last_reward = None
                last_info = None
                print("\nEpisodio reiniciado")

    if not done and current_time - last_action_time >= action_delay_ms:
        action = agent.select_action(state, epsilon=0.0)

        next_state, reward, done, info = env.step(action)

        total_reward += reward
        step_count += 1

        last_action = action
        last_reward = reward
        last_info = info

        print(
            f"Paso {step_count} | "
            f"Acción: {ACTION_NAMES[action]} | "
            f"Reward: {reward} | "
            f"Posición: {env.agent_pos} | "
            f"Info: {info}"
        )

        state = next_state
        last_action_time = current_time

        if done:
            print("\nEpisodio terminado")
            print("Total reward:", total_reward)
            print("Pasos:", step_count)

    renderer.draw()

pygame.quit()