import pickle
import numpy as np
from TrainingEnvironment import TrainingEnvironment
import csv
from QPlayer import QPlayer

PICKLE = "./pickles/pickle3.25.pickle"
LOGS = "./logs/pickle_test_3.23.csv"
TEST_SIZE = 1000000 # number of games to play

with open(PICKLE, "rb") as file:
    Q = pickle.load(file)
    #Q = np.load(file, allow_pickle=True)

wins = 0
loses = 0
env = TrainingEnvironment()
logs = []
move_dist = [0,0,0,0,0,0,0,0,0,0,0,0,0]

for episode in range(TEST_SIZE):
    obs, reward, done = env.reset()
    moves = 0

    while not done:

        hs = QPlayer.hash(obs)

        action = np.argmax(Q[hs])
        move_dist[action] += 1

        obs, reward, done = env.move(action)
        
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
    
    logs.append([episode, wins, loses, wins / (wins + loses) if wins + loses > 0 else 0])

#with open(LOGS, 'w', newline='') as file:
#    csvwriter = csv.writer(file)
#    csvwriter.writerows(logs)

print("Over 100000 episodes there were {} wins and {} loses for a win percentage of {}".format(wins, loses, wins/(wins+loses)))
print(move_dist, sum(move_dist))
