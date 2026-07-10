import torch

from env.maps import SIMPLE_MAZE, SIMPLE_START, SIMPLE_GOAL
from env.maze_env import MazeEnv
from agents.dqn_agent import DQNAgent
from memory.replay_buffer import ReplayBuffer


def get_target_q_values(agent, state):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        q_values = agent.target_q_network(state_tensor)

    return q_values.squeeze(0)


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


test_state = env.reset()

main_before = agent.get_q_values(test_state)
target_before = get_target_q_values(agent, test_state)

print("Q-network inicial:")
print(main_before)

print("\nTarget network inicial:")
print(target_before)

print("\n¿Son iguales al inicio?")
print(torch.allclose(main_before, target_before))


loss = agent.learn(
    memory=memory,
    batch_size=3,
    gamma=0.99,
)

main_after_learn = agent.get_q_values(test_state)
target_after_learn = get_target_q_values(agent, test_state)

difference_after_learn = torch.abs(main_after_learn - target_after_learn).sum().item()

print("\nLoss después de aprender:")
print(loss)

print("\nQ-network después de learn:")
print(main_after_learn)

print("\nTarget network después de learn:")
print(target_after_learn)

print("\nDiferencia después de learn:")
print(difference_after_learn)

print("\n¿Son iguales después de learn?")
print(torch.allclose(main_after_learn, target_after_learn))


agent.update_target_network()

main_after_update = agent.get_q_values(test_state)
target_after_update = get_target_q_values(agent, test_state)

difference_after_update = torch.abs(main_after_update - target_after_update).sum().item()

print("\nDespués de update_target_network()")

print("\nQ-network:")
print(main_after_update)

print("\nTarget network:")
print(target_after_update)

print("\nDiferencia después de update:")
print(difference_after_update)

print("\n¿Son iguales después de update?")
print(torch.allclose(main_after_update, target_after_update))