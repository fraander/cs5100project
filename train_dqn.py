import random
import numpy as np
import torch
from collections import deque
from datetime import datetime
import csv

from dqn_env import TrainingDQNEnv, ACTION_LIST
from dqn import DQN

LOGS = "./logs/logs4.8.csv"

def train_dqn(num_episodes=5000,
              batch_size=100,
              gamma=0.95,
              epsilon_start=1.0,
              epsilon_end=0.01,
              epsilon_decay=0.9999999,
              replay_size=1000,
              logs=100,
              target_update_freq=500):

    env = TrainingDQNEnv(num_players=3)
    input_dim = 242
    output_dim = len(ACTION_LIST)

    dqn = DQN(input_dim=input_dim, output_dim=output_dim, lr=0.0001)
    # Target DQN
    dqn_target = DQN(input_dim=input_dim, output_dim=output_dim, lr=0.0001)
    dqn_target.load_state_dict(dqn.state_dict())  # copy weights initially

    replay_buffer = deque(maxlen=replay_size)
    logging_data = []

    epsilon = epsilon_start

    wins = 0
    losses = 0
    total_loss = 0.0
    update_count = 0
    total_rewards = []

    for episode in range(num_episodes):
        obs, _, done = env.reset()
        inputs = env.get_inputs()  # shape (242,)
        step = 0
        max_steps = 500

        while not done and step < max_steps:
            step += 1

            # Epsilon-greedy with legal action masking
            if random.random() < epsilon:
                # Exploration: pick a random action **among the legal moves** only
                legal_actions = env.get_legal_actions()
                legal_indices = [i for i, act in enumerate(ACTION_LIST) if act in legal_actions]
                action_idx = random.choice(legal_indices)
            else:
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
            next_obs, reward, done = env.step(action_str)
            next_inputs = env.get_inputs()

            # Store transition in replay buffer
            replay_buffer.append((inputs, action_idx, reward, next_inputs, done))

            inputs = next_inputs
            # Training step if buffer is big enough
            if episode > 5 and len(replay_buffer) >= batch_size:
                update_count += 1
                batch = random.sample(replay_buffer, batch_size)

                s_list, a_list, r_list, s_next_list, d_list = [], [], [], [], []
                for (ss, aa, rr, sn, dd) in batch:
                    s_list.append(ss)
                    a_list.append(aa)
                    r_list.append(rr)
                    s_next_list.append(sn)
                    d_list.append(dd)

                s_batch_t = torch.from_numpy(np.array(s_list, dtype=np.float32))
                a_batch_t = torch.from_numpy(np.array(a_list, dtype=np.int64)).unsqueeze(1)
                r_batch_t = torch.from_numpy(np.array(r_list, dtype=np.float32)).unsqueeze(1)
                s_next_batch_t = torch.from_numpy(np.array(s_next_list, dtype=np.float32))
                d_batch_t = torch.from_numpy(np.array(d_list, dtype=np.float32)).unsqueeze(1)

                # Current Q(s,a) from online DQN
                q_values_batch = dqn(s_batch_t)  # [B, 61]
                q_sa = q_values_batch.gather(1, a_batch_t)  # [B,1]

                # Next Q from target DQN
                with torch.no_grad():
                    q_next_target = dqn_target(s_next_batch_t)  # [B, 61]
                    max_q_next = q_next_target.max(dim=1, keepdim=True)[0]  # [B,1]

                target = r_batch_t + gamma * max_q_next * (1 - d_batch_t)

                loss = dqn.compute_loss(q_sa, target)
                dqn.update_params(clip_value=1.0)
                total_loss += loss.item()

                # Update target network occasionally
                if update_count % target_update_freq == 0:
                    dqn_target.load_state_dict(dqn.state_dict())

            # Decay epsilon
            if epsilon > epsilon_end:
                epsilon *= epsilon_decay

            # Track wins/losses
            if reward != 0:
                if reward > 0:
                    wins += 1
                else:
                    losses += 1
                total_rewards.append(reward)



        # End of episode logging
        if (episode + 1) % logs == 0:
            avg_reward = np.mean(total_rewards)  # average last N steps in this episode
            avg_loss = total_loss / logs
            avg_win = wins / (wins + losses) if (wins + losses) > 0 else 0

            total_loss = 0.0
            wins = 0
            losses = 0
            total_rewards = []

            logging_data.append([datetime.now(), episode, epsilon, avg_loss, avg_win, avg_reward])
            with open(LOGS, 'w', newline='') as file:
                csvwriter = csv.writer(file)
                csvwriter.writerows(logging_data)

            print(f"Episode {episode + 1} | Epsilon {epsilon:.4f} | Avg Loss {avg_loss:.4f} | Win Rate {avg_win:.4f} | Avg Reward {avg_reward:.4f}")

    # Save model
    torch.save(dqn.state_dict(), "dqn_save.pth")
    print("Training complete. Model saved to 'dqn_sample.pth'")

if __name__ == "__main__":
    train_dqn(num_episodes=100000)
