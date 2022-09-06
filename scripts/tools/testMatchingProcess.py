#!/usr/bin/python3

import os
import sys
import inspect
import torch
import torch.nn as nn

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from typing import List
from lv import LV, Params
from math import inf
from lvAnalyzer import Analyzer
import json
import numpy as np

def loadModel(file: str):
    model = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 1), nn.Sigmoid())
    # model = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 64), nn.Linear(64, 1), nn.Sigmoid())
    model.load_state_dict(torch.load(file))
    model.eval()
    return model

dataset = []
model = loadModel(os.path.dirname(os.path.realpath(__file__)) + "/../model/modelNew1.pth")

# with open("../datasets/scenesWarehouseWithFeatures.json", "r") as f:
#     dataset = json.load(f)


with open("../datasets/scenesSmallHouseWithFeatures.json", "r") as f:
    dataset = json.load(f)

# with open("../datasets/scenesHospitalWithFeatures.json", "r") as f:
#     dataset = json.load(f)

#threshold=0.73714
paramsArray=Params([
    1.58773, 22.3013, 0.0406,
    0.221766, 4.691886, 0.82588,
    0.026518, 15.95429, 0.058236,
    11.919109, 7.489252, 0.79746627
])

# th -- 0.646

def test(th1, th2):
    global dataset
    global model
    lvs = []
    anal = Analyzer(False)
    anal = Analyzer(storeImages=False, fpPosThreshold=0.8, fpRotThreshold=0.4, complexAnal=True)
    for d in dataset:
        lv = LV.parseLV(d["scene"], paramsArray, d["features"])
        bestViewIndex = -1
        bestSimilarity = -inf
        for (i, otherView) in enumerate(lvs):
            similarity = lv.match(otherView, 0.84, th2, model)
            if similarity > bestSimilarity:
                bestViewIndex = i
                bestSimilarity = similarity
        if bestSimilarity < th1:
            bestViewIndex = len(lvs)
            lvs.append(lv)
        anal.insert(bestViewIndex, d["position"])
    return (anal.falsePositives, anal.falseNegatives, len(anal.all.keys()), anal.totalReceived, th1)



# for t1 in list(np.linspace(0.4,0.8,14)):
#     res = []
#     for t2 in list(np.linspace(0, 1, 101)):
#         res.append(test(t1, t2))
#         print(str(t1) + " - " + str(t2))
#     print(res)

#     with open("output" + str(t1) + ".json", "w") as f:
#         json.dump(res, f)


res = []
for t1 in list(np.linspace(0, 1, 101)):
    actRes = test(t1, 0.00134)
    res.append(actRes)
    print(str(t1) + ": " + str(actRes))
with open("output.json", "w") as f:
    json.dump(res, f)
print(res)


