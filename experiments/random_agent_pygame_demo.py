import pygame

from env.maps import SIMPLE_MAZE, SIMPLE_START, SIMPLE_GOAL
from env.maze_env import MazeEnv
from env.renderer import MazeRenderer
from agents.random_agent import RandomAgent


env = MazeEnv(
    maze=SIMPLE_MAZE,
    start_pos=SIMPLE_START,
    goal_pos=SIMPLE_GOAL,
    max_steps=50,
)

agent = RandomAgent(num_actions=4)
renderer = MazeRenderer(env, cell_size=80)

state = env.reset()
done = False
running = True

last_action_time = 0
action_delay_ms = 800

while running:
    current_time = pygame.time.get_ticks()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                state = env.reset()
                done = False
                print("Episodio reiniciado.")
                
            if event.key == pygame.K_ESCAPE:
                running = False

    if not done and current_time - last_action_time >= action_delay_ms:
        action = agent.select_action(state)

        next_state, reward, done, info = env.step(action)

        print("-" * 40)
        print("Acción:", action)
        print("Posición:", env.agent_pos)
        print("Recompensa:", reward)
        print("Done:", done)
        print("Info:", info)

        state = next_state
        last_action_time = current_time

    renderer.draw()
    renderer.tick(fps=30)

renderer.close()