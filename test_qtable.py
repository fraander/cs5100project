import pickle
import numpy as np
from TrainingEnvironment import TrainingEnvironment
import csv
from QPlayer import QPlayer

PICKLE = "./pickles/pickle3.25.pickle"
LOGS = "./logs/pickle_test_3.23.csv"
TEST_SIZE = 1000000 # number of games to play

# Open the chosen Pickle to run the test with
with open(PICKLE, "rb") as file:
    Q = pickle.load(file)

# Setup metrics for tracking
wins = 0
loses = 0
env = TrainingEnvironment()
logs = []
move_dist = [0,0,0,0,0,0,0,0,0,0,0,0,0]

# Run through the chosen number of games for the test
for episode in range(TEST_SIZE):
    # Start a new game
    obs, reward, done = env.reset()
    moves = 0

    # Play the game to completion
    while not done:

        # Choose the best move from the table
        hs = QPlayer.hash(obs)
        action = np.argmax(Q[hs])
        move_dist[action] += 1

        # Make the move
        obs, reward, done = env.move(action)
        
        # Update the metrics as relevant
        if reward == TrainingEnvironment.rewards['wrong_card']:
            print(obs['current_card'], action)
            done = True
            loses += 1
        elif reward == TrainingEnvironment.rewards['lose']:
            done = True
            loses += 1
        elif reward == TrainingEnvironment.rewards['win']:
            done = True
            wins += 1
    
    # Add the result to logging
    logs.append([episode, wins, loses, wins / (wins + loses) if wins + loses > 0 else 0])


print("Over 100000 episodes there were {} wins and {} loses for a win percentage of {}".format(wins, loses, wins/(wins+loses)))
print(move_dist, sum(move_dist))
