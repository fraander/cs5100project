from dqn_env import TrainingDQNEnv, ACTION_LIST
import torch
import numpy as np
from dqn import DQN

def test_agent(dqn, num_episodes=100, epsilon=0.0):

    env = TrainingDQNEnv(num_players=3)

    wins = 0
    losses = 0
    for ep in range(num_episodes):
        obs_dict, _, done = env.reset()
        inputs = env.get_inputs()
        episode_reward = 0

        while not done:
            # Epsilon-greedy, with epsilon=0 => always pick best
            with torch.no_grad():
                s_t = torch.FloatTensor(inputs).unsqueeze(0)  # [1, 242]
                q_values = dqn(s_t)

                # Build a legal mask
                legal_actions = env.get_legal_actions()
                legal_mask = torch.tensor(
                    [[act in legal_actions for act in ACTION_LIST]],
                    dtype=torch.bool
                )
                q_values_masked = q_values.clone()
                q_values_masked[~legal_mask] = -1e9  # mask out illegal moves
                action_idx = q_values_masked.argmax(dim=1).item()

            action_str = ACTION_LIST[action_idx]
            next_obs_dict, reward, done = env.step(action_str)
            inputs = env.get_inputs()
            episode_reward += reward

        if episode_reward > 0:
            wins += 1
        if episode_reward < 0:
            losses += 1

    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    print(f"Tested over {num_episodes} episodes. Win Rate = {win_rate:.2f}")
    return win_rate

if __name__ == "__main__":

    dqn_test = DQN(242, 61, lr=0.0001)
    dqn_test.load_state_dict(torch.load("dqn_sample.pth"))
    dqn_test.eval()

    results = []

    # Run the test 100 times
    for run in range(100):
        result = test_agent(dqn_test, num_episodes=1000, epsilon=0.0)
        results.append(result)
        print(f"Run {run+1}: {result}")

    # Calculate and print the mean result over 100 runs
    mean_result = np.mean(results)
    print("Mean result over 100 runs:", mean_result)
