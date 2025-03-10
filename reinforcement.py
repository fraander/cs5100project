import time
import pickle
import numpy as np
from TrainingEnvironment import TrainingEnvironment

env = TrainingEnvironment()
NUM_ACTIONS = 9

def hash(obs):
    
    match_num = any([c.card_type == obs['current_card'].card_type for c in obs['hand']])
    match_color = any([c.color == obs['current_card'].color for c in obs['hand']])
    match_both = any([c.color == obs['current_card'].color and c.card_type == obs['current_card'].card_type for c in obs['hand']])
    black_card = any([c.color == 'black' for c in obs['hand']])
    skip_card = any([c.card_type == 'skip' for c in obs['hand']])
    rev_card = any([c.card_type == 'reverse' for c in obs['hand']])

    return match_num*32 + match_color*16 + match_both*8 + black_card*4 + skip_card*2 + rev_card

def test_table(Q_table, num_episodes):
    
    avg_reward = 0
    wins = 0
    loses = 0
    
    for _ in range(num_episodes):
        
        obs, reward, done = env.reset()
        steps = 0

        while done == False and steps < 100:
            hs = hash(obs)

            action = np.argmax(Q_table[hs])
            obs, reward, done = env.move(action)
            steps += 1

            avg_reward += reward
        
        if reward == 10000:
            wins += 1
        elif reward == -10000:
            loses += 1
            
    return avg_reward / num_episodes, wins, loses

def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):

    Q_table = {}
    for i in range(375):
        Q_table[i] = np.zeros(NUM_ACTIONS)
    num_updates = np.zeros((375,NUM_ACTIONS))

    for episode in range(num_episodes):

        if episode % 10000 == 0:
            print("{} out of {} episodes. The Q table has {} entries, the exploration rate is {}".format(episode, num_episodes, len(Q_table.keys()), epsilon))
            avg, wins, loses = test_table(Q_table, 1000)
            print("The Q_Table got an average reward of {} with {} wins and {} loses for a win percentage of {}".format(avg, wins, loses, wins / (wins + loses) if wins + loses > 0 else 0))
        
        obs, reward, done = env.reset()
        moves = 0

        while done == False:
            hs = hash(obs)

            if np.random.random() > epsilon:
                action = np.argmax(Q_table[hs])
            else:
                action = np.random.randint(0,NUM_ACTIONS)

            obs, reward, done = env.move(action)
            moves += 1

            eta = 1/(1+num_updates[hs, action])
            newQ = ((1-eta)*Q_table[hs][action]) + (eta*(reward+(gamma * np.max(Q_table[hash(obs)]))))

            Q_table[hs][action] = newQ
            num_updates[hs, action] += 1
        
        epsilon *= decay_rate
        #print("That game had {} steps and ended with result {}".format(moves, reward))
    
    return Q_table

decay_rate = 0.999995

Q_table = Q_learning(num_episodes=1000000, gamma=0.9, epsilon=1, decay_rate=decay_rate) # Run Q-learning
# Q_table = Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=decay_rate) # Run Q-learning

# Save the Q-table dict to a file
with open('Q_table.pickle', 'wb') as handle:
    pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)


'''
Uncomment the code below to play an episode using the saved Q-table. Useful for debugging/visualization.

Comment before final submission or autograder may fail.
'''

# Q_table = np.load('Q_table.pickle', allow_pickle=True)

# obs, reward, done, info = env.reset()
# total_reward = 0
# while not done:
# 	state = hash(obs)
# 	action = np.argmax(Q_table[state])
# 	obs, reward, done, info = env.step(action)
# 	total_reward += reward
# 	if gui_flag:
# 		refresh(obs, reward, done, info)  # Update the game screen [GUI only]

# print("Total reward:", total_reward)

# # Close the
# env.close() # Close the environment