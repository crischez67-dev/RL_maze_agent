from env.maps import SIMPLE_MAZE, SIMPLE_START, SIMPLE_GOAL
from env.maze_env import MazeEnv
from memory.replay_buffer import ReplayBuffer

from models.q_network import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim


env = MazeEnv(
    maze=SIMPLE_MAZE,
    start_pos=SIMPLE_START,
    goal_pos=SIMPLE_GOAL,
    max_steps=50,
)

buffer = ReplayBuffer(capacity=100)

state = env.reset()

actions = [3, 3, 3, 3, 1, 1, 1]

for action in actions:
    next_state, reward, done, info = env.step(action)

    buffer.push(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=done,
    )

    state = next_state

    if done:
        break

print("Tamaño del buffer:", len(buffer))

batch = buffer.sample(batch_size=3)

print("\nBatch aleatorio:")
for experience in batch:
    print(experience)

states, actions, rewards, next_states, dones = buffer.sample_tensors(batch_size=3)

print("\nTensores del batch:")
print("states:", states)
print("states shape:", states.shape)

print("actions:", actions)
print("actions shape:", actions.shape)

print("rewards:", rewards)
print("rewards shape:", rewards.shape)

print("next_states:", next_states)
print("next_states shape:", next_states.shape)

print("dones:", dones)
print("dones shape:", dones.shape)

q_network = QNetwork(
    state_size=6,
    action_size=4,
)

optimizer = optim.Adam(q_network.parameters(), lr=0.001)

q_values = q_network(states)

print("\nValores Q del batch:")
print(q_values)
print("q_values shape:", q_values.shape)

selected_q_values = q_values.gather(1, actions)

print("\nValores Q seleccionados:")
print(selected_q_values)
print("selected_q_values shape:", selected_q_values.shape)

gamma = 0.99

with torch.no_grad():
    next_q_values = q_network(next_states)
    max_next_q_values = next_q_values.max(dim=1, keepdim=True)[0]
    target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

print("\nValores Q de los siguientes estados:")
print(next_q_values)
print("next_q_values shape:", next_q_values.shape)

print("\nMáximos Q de los siguientes estados:")
print(max_next_q_values)
print("max_next_q_values shape:", max_next_q_values.shape)

print("\nValores Q objetivo:")
print(target_q_values)
print("target_q_values shape:", target_q_values.shape)

loss = F.mse_loss(selected_q_values, target_q_values)

print("\nLoss:")
print(loss)
print("loss shape:", loss.shape)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("\nSe realizó un paso de actualización de la red.")

new_q_values = q_network(states)
new_selected_q_values = new_q_values.gather(1, actions)

new_loss = F.mse_loss(new_selected_q_values, target_q_values)

print("\nLoss después de la actualización:")
print(new_loss)