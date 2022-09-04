import json
import random
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from lv import LV, Params
import numpy as np
import matplotlib.pyplot as plt



def createMLDataset(inputFile: str, outputFile: str):
    data = []
    posThreshold = 0.5
    rotThreshold = 0.2
    with open(inputFile, "r") as f:
        data = json.load(f)

    print(len(data))
    pairedData = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            pairedData.append({"T1": data[i], "T2": data[j]})

    finalData = []
    for d in pairedData:
        t1 = d["T1"]
        t2 = d["T2"]
        x,y,a = t1["position"]
        mx,my,ma = t2["position"]
        isPositive = (mx - x)**2 + (my - y) ** 2 < posThreshold ** 2 and min(abs(a - ma), abs(abs(a - ma) - 2)) < rotThreshold
        finalData.append({"d": t1["features"] + t2["features"], "c": 1 if isPositive else 0})



    positives = [x for x in finalData if x["c"] == 1][:5000]
    negatives = [x for x in finalData if x["c"] == 0][:5000]
    result = positives + negatives
    random.shuffle(result)
    print(len(data))
    print(len(result))
    with open(outputFile, "w") as f:
        json.dump(result, f)

createMLDataset("../datasets/scenesWarehouseWithFeatures.json", "../datasets/learningDataset.json")

def pairAndClassifyDataset(inputFile: str, outputFile: str):
    data = []
    posThreshold = 0.5
    rotThreshold = 0.2
    params=Params([
        6.89214682e+00, 1.80208393e+01, 1.49458688e-01,
        6.44444083e-02, 1.74226278e+01, 9.28374238e-01,
        5.41219509e-01, 2.66566342e+00, 1.33238250e-02,
        1.72236808e+01, 1.47960297e+01, 2.48803591e-01
    ])
    threshold=3.80785060e-01

    with open(inputFile, "r") as f:
        data = json.load(f)
    pairedData = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            print(str(i * len(data) + j) + " / " + str(len(data) * (len(data) - 1) / 2))
            if LV.parseLV(data[j]["scene"], params).match(LV.parseLV(data[i]["scene"], params)) > threshold:
                pairedData.append({"T1": data[i], "T2": data[j]})
    finalData = []
    for d in pairedData:
        t1 = d["T1"]
        t2 = d["T2"]
        x,y,a = t1["position"]
        mx,my,ma = t2["position"]
        isPositive = (mx - x)**2 + (my - y) ** 2 < posThreshold ** 2 and min(abs(a - ma), abs(abs(a - ma) - 2)) < rotThreshold
        finalData.append({"d": t1["features"] + t2["features"], "c": 1 if isPositive else 0})
    random.shuffle(finalData)
    print(len(data))
    print(len(finalData))
    with open(outputFile, "w") as f:
        json.dump(finalData, f)

def pairAndClassifyAll(inputFile: str, outputFile: str):
    data = []
    posThreshold = 0.5
    rotThreshold = 0.2
    params=Params([
        6.89214682e+00, 1.80208393e+01, 1.49458688e-01,
        6.44444083e-02, 1.74226278e+01, 9.28374238e-01,
        5.41219509e-01, 2.66566342e+00, 1.33238250e-02,
        1.72236808e+01, 1.47960297e+01, 2.48803591e-01
    ])
    threshold=3.80785060e-01

    with open(inputFile, "r") as f:
        data = json.load(f)
    pairedData = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if random.random() < 0.1:
                pairedData.append({"T1": data[i], "T2": data[j]})
    finalData = []
    for d in pairedData:
        t1 = d["T1"]
        t2 = d["T2"]
        x,y,a = t1["position"]
        mx,my,ma = t2["position"]
        isPositive = (mx - x)**2 + (my - y) ** 2 < posThreshold ** 2 and min(abs(a - ma), abs(abs(a - ma) - 2)) < rotThreshold
        finalData.append({"d": t1["features"] + t2["features"], "c": 1 if isPositive else 0})
    random.shuffle(finalData)
    print(len(data))
    print(len(finalData))
    with open(outputFile, "w") as f:
        json.dump(finalData, f)

def visualizeDataset(fileName):
    data = []
    with open(fileName, "r") as f:
        data = json.load(f)
    for d in data:
        first = d["d"][:1024]
        second = d["d"][1024:]
        classValue = d["c"]
        diffs = list(np.array(first) - np.array(second))
        plt.hist(diffs)
        plt.title("SAME" if classValue == 1 else "DIFFERENT")
        plt.xlim(-200, 200)
        plt.show()
