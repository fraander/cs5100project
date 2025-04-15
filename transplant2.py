import pickle
import numpy as np
from TrainingEnvironment import TrainingEnvironment
import csv

NPPICKLE = "./pickle_final.pickle"
SAVE = "./pickle_final.csv"

with open(SAVE) as file:
    t = csv.reader(file)

    Q = {}
    for i,n in enumerate(t):
        Q[i] = [n]

with open(NPPICKLE, 'wb') as handle:
    pickle.dump(Q, handle, protocol=pickle.HIGHEST_PROTOCOL)