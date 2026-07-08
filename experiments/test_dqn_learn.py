from env.maps import SIMPLE_MAZE, SIMPLE_START, SIMPLE_GOAL
from env.maze_env import MazeEnv
from agents.dqn_agent import DQNAgent
from memory.replay_buffer import ReplayBuffer


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
    learning_rate=0.001,
)

memory = ReplayBuffer(capacity=100)

state = env.reset()

actions = [3, 3, 3, 3, 1, 1, 1]

for action in actions:
    next_state, reward, done, info = env.step(action)

    memory.push(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=done,
    )

    state = next_state

    if done:
        break

print("Experiencias en memoria:", len(memory))

loss = agent.learn(
    memory=memory,
    batch_size=3,
    gamma=0.99,
)

print("Loss devuelta por agent.learn():", loss)