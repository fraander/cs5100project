import pickle
import numpy as np
from TrainingEnvironment import TrainingEnvironment
import csv

PICKLE = "./pickles/pickle3.25.pickle"
SAVE = "./pickles/pickle3.25.csv"

with open(PICKLE, "rb") as file:
    Q = pickle.load(file)

t = []
for i in range(len(Q)):
    t.append(Q[i])

with open(SAVE, 'w', newline='') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerows(t)