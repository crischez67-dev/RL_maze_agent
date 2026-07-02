import csv
import os

def run_episode(env, agent):
    state = env.reset()

    done = False
    step_count = 0
    total_reward = 0
    final_info = None

    while not done:
        action = agent.select_action(state)

        next_state, reward, done, info = env.step(action)

        step_count += 1
        total_reward += reward

        state = next_state
        final_info = info

    final_position = env.agent_pos

    return step_count, total_reward, final_info, final_position

def evaluate_agent(env, agent, output_file, num_episodes=100):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    total_steps_all = 0
    total_reward_all = 0

    collisions = 0
    goals = 0
    timeouts = 0

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "episode",
            "steps",
            "total_reward",
            "collision",
            "goal",
            "timeout",
            "final_row",
            "final_col",
        ])

        for episode in range(1, num_episodes + 1):
            steps, total_reward, info, final_position = run_episode(env, agent)

            final_row, final_col = final_position

            writer.writerow([
                episode,
                steps,
                total_reward,
                info["collision"],
                info["goal"],
                info["timeout"],
                final_row,
                final_col,
            ])

            total_steps_all += steps
            total_reward_all += total_reward

            if info["collision"]:
                collisions += 1

            if info["goal"]:
                goals += 1

            if info["timeout"]:
                timeouts += 1

    summary = {
        "episodes": num_episodes,
        "collisions": collisions,
        "goals": goals,
        "timeouts": timeouts,
        "average_steps": total_steps_all / num_episodes,
        "average_reward": total_reward_all / num_episodes,
    }

    return summary