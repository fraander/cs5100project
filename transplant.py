import pickle
import numpy as np
from TrainingEnvironment import TrainingEnvironment
import csv

PICKLE = "./pickles/pickle3.25testing.pickle"
SAVE = "./pickle_final.csv"

with open(PICKLE, "rb") as file:
    Q = pickle.load(file)

t = []
for i in range(len(Q)):
    t.append([float(i) for i in Q[i]])

with open(SAVE, 'w', newline='') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerows(t)