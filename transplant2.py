import pickle
import numpy as np
from TrainingEnvironment import TrainingEnvironment
import csv

NPPICKLE = "./pickles/pickle3.25.normal.pickle"
SAVE = "./pickles/pickle3.25.csv"

with open(SAVE) as file:
    t = csv.reader(file)

    Q = {}
    for i,n in enumerate(t):
        Q[i] = [n]

with open(NPPICKLE, 'wb') as handle:
    pickle.dump(Q, handle, protocol=pickle.HIGHEST_PROTOCOL)