import pickle
from datetime import datetime, timedelta
import csv
import numpy as np
from QPlayer import QPlayer

from TrainingEnvironment import TrainingEnvironment

env = TrainingEnvironment()
NUM_ACTIONS = 13
NUM_STATES = 76801

# !! Configuration !!
LOGS = "./logs/logs4.4.csv"
PICKLE = "./pickles/pickle4.4.pickle"
RUNTIME = 750 # measured in minutes

# Open the Q-table from a file
def read_q(file_path=None):
    if file_path is not None:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    return None

def test_table(q, num_episodes=1000):
    """
    Runs a set number of games, calculates win-loss record
    """
    # Metrics to track
    avg_reward = 0
    wins = 0
    loses = 0
    illegal_moves = 0
    legal_moves = 0

    # Run each of the games
    for _ in range(num_episodes):

        obs, reward, done = env.reset()
        steps = 0

        # Play game until completion; or 100 played moves
        while done == False and steps < 100:

            # Get the best action for the given state
            hs = QPlayer.hash(obs)

            action = np.argmax(q[hs])
            
            # Take the action
            obs, reward, done = env.move(action)
            
            # Adjust metrics as relevant
            steps += 1
            avg_reward += reward
            if reward == TrainingEnvironment.rewards['wrong_card']:
                illegal_moves += 1
                loses += 1
                done = True
            else:
                legal_moves += 1

        if reward == TrainingEnvironment.rewards['win']:
            wins += 1
        elif reward == TrainingEnvironment.rewards['lose']:
            loses += 1

    # Share result of test run
    print("The table made {} legal and {} illegal moves".format(legal_moves, illegal_moves))
    return avg_reward / num_episodes, wins, loses, legal_moves, illegal_moves


def avg_score(Q):
    """
    Calculate the average value of all cells in the Q table 
    """

    actions = [0] * NUM_ACTIONS
    taz = 0
    for i in Q.keys():
        all_zero = True
        for action in range(NUM_ACTIONS):
            actions[action] += Q[i][action]
            if Q[i][action] != 0:
                all_zero = False
        taz += 1 if all_zero else 0
    return [round(a/len(Q.keys()),2) for a in actions], taz

def Q_learning(gamma=0.9, epsilon=1, decay=0.999, q_path=None):
    """
    Train the Agent by populating and optimizing the Q-table. Uses Epsilon-Greedy.
    """

    # Tracking metrics
    start = datetime.now()
    checkpoint = datetime.now()
    logging = []
    unique_states = set()
    episode = 0

    # Initialize the Q- and num-updates tables using a given path, if provided. Otherwise, as default
    loaded = read_q(q_path)
    Q = loaded if loaded is not None else {}
    if loaded is None:
        for i in range(NUM_STATES):
            Q[i] = np.zeros(NUM_ACTIONS)
    num_updates = np.zeros((NUM_STATES, NUM_ACTIONS))

    # Run for the noted period of time
    while datetime.now() < start + timedelta(minutes=RUNTIME):
        
        # If at a 5-minute interval, perform a test to see progress
        if datetime.now() > checkpoint + timedelta(minutes=5):
            print("{} out of ?? episodes. The Q table has {} entries and has seen {} unique states, the exploration rate is {}".format(episode,
                                                                                                                                       len(Q.keys()),
                                                                                                                                       len(unique_states),
                                                                                                                                       epsilon))
            avg, wins, loses, legal, illegal = test_table(Q, 10000)
            print("The Q_Table got an average reward of {} with {} wins and {} loses for a win percentage of {}".format(
                avg, wins, loses, wins / (wins + loses) if wins + loses > 0 else 0))
            avg_scores, taz = avg_score(Q)
            print("Average Q score for each action: {}. There are {} states with no scores".format(avg_scores, taz))
            print()
            logging.append([datetime.now(), episode, epsilon, len(unique_states), legal, illegal, legal / (legal + illegal) if legal + illegal > 0 else 0, avg, wins, loses, wins / (wins + loses) if wins + loses > 0 else 0, taz])
            for i in avg_scores:
                logging[-1].append(i)
            checkpoint = datetime.now()
            epsilon *= decay
            with open(LOGS, 'w', newline='') as file:
                csvwriter = csv.writer(file)
                csvwriter.writerows(logging)

            # Save the Q-table dict to a file
            with open(PICKLE, 'wb') as handle:
                pickle.dump(Q, handle, protocol=pickle.HIGHEST_PROTOCOL)

        obs, reward, done = env.reset()
        moves = 0

        # Play a game to completion
        while not done:

            hs = QPlayer.hash(obs)

            unique_states.add(hs)

            if np.random.random() > epsilon:
                action = np.argmax(Q[hs])
            else:
                action = np.random.randint(0, NUM_ACTIONS)

            obs, reward, done = env.move(action)
            moves += 1

            eta = 1 / (1 + num_updates[hs, action])
            newQ = ((1 - eta) * Q[hs][action]) + (eta * (reward + (gamma * np.max(Q[QPlayer.hash(obs)]))))

            Q[hs][action] = newQ
            num_updates[hs, action] += 1

        episode += 1
        #print("That game had {} steps and ended with result {}".format(moves, reward))

    return Q, logging


decay_rate = 0.99

# Give the file path of the Q_table.pickle to load an existing Q_table
Q_table, logs = Q_learning(gamma=0.9, epsilon=1, decay=decay_rate, q_path=None)  # Run Q-learning
# Q_table = Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=decay_rate) # Run Q-learning

with open(LOGS, 'w', newline='') as file:
    csvwriter = csv.writer(file)
    csvwriter.writerows(logs)

# Save the Q-table dict to a file
with open(PICKLE, 'wb') as handle:
    pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

