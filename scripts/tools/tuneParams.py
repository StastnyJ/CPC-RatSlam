#!/usr/bin/python3

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from typing import List
from lv import LV, Params
from math import inf
from lvAnalyzer import Analyzer
import json
from datetime import datetime

from scipy.optimize import differential_evolution

bounds = [
    (0.0,15.0), (0.0, 25.0), (0.0, 1.0),
    (0.0,1.0), (0.0, 25.0), (0.0, 1.0),
    (0.0, 1.0), (0.0, 25.0), (0.0, 1.0),
    (0.0, 20.0), (0.0, 25.0), (0.0, 1.0),
    (0.0, 1.0)
]

##
#   [1.3412207,  11.51117396,  0.07464632,  0.04870507,  8.427029,    0.93453884,  0.72795924,  6.07578882,  0.09846648,  6.37736941,  4.5353558,   0.27635345,  0.51402502]
#   
##

dataset = []
with open("../datasets/scenesWarehouse.json", "r") as f:
    dataset = json.load(f)#[100:250]

def fitness(rawParams: List[float]):
    global dataset
    params = Params(rawParams)
    lvs = []
    anal = Analyzer(False)
    for d in dataset:
        lv = LV.parseLV(d["scene"], params)
        bestViewIndex = -1
        bestSimilarity = -inf
        for (i, otherView) in enumerate(lvs):
            similarity = lv.match(otherView)
            if similarity > bestSimilarity:
                bestViewIndex = i
                bestSimilarity = similarity
        if bestSimilarity < rawParams[-1]:
            bestViewIndex = len(lvs)
            lvs.append(lv)
        anal.insert(bestViewIndex, d["position"])
    return anal.falsePositives + anal.falseNegatives

start = datetime.now()
result = differential_evolution(fitness, bounds, workers=-1, callback=lambda x, convergence: print(str(x) + "; " + str(convergence) + " --------- " + str(fitness(x))))
