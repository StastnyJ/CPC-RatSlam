#!/usr/bin/python3

import plotly.express as px
import itertools
from math import sqrt
import json


def makeFpDetailsGraph():
    rawData = []

    with open(analFolder + "fpDetails.txt", "r") as f:
        rawData = [tuple([float(x) for x in row[:-1].split(";")]) for row in f.readlines()]

    data = list(itertools.chain(*[
        [
            {"id": i + 1, "actual": sqrt(d[0]), "category": "current"},
            {"id": i + 1, "actual": sqrt(d[2]), "category": "best"}, 
        ][:1 if d[0] == d[2] else 2] for (i,d) in enumerate(rawData)
    ]))

    fig = px.scatter(data, x="id", y="actual", color="category")
    fig.add_hline(y=0.8)
    fig.show()

def makePRCurve(files, curveNames):
    def parseRow(raw: str):
        fp = raw[0]
        fn = raw[1]
        tn = raw[2] - fn
        tp = raw[3] - fp - fn - tn 
        t = raw[-1]
        return [tp, tn, fp, fn, t]

    data = []
    for (i, fn) in enumerate(files):
        rawData = []
        with open(fn, "r") as f:
            rawData = [parseRow(x) for x in json.load(f)]

        data += [{
            'threshold': d[4],
            'accuracy': sum(d[:2]) / sum(d),
            'precission': (d[0] + 0.00000001) / ((d[0] + d[2]) + 0.00000001),
            'recall': (d[0] + 0.00000001) / (d[0] + d[3] + 0.00000001),
            "category": curveNames[i]
        } for d in rawData]
    
    fig = px.line(data, y="precission",x="recall", color="category", markers=True)
    fig.show()

    fig2 = px.line(data, x="threshold", y="accuracy", color="category", markers=True)
    fig2.show()

makePRCurve([
        "../tests/1stStageOnly.json",
        "../tests/2ndStageOnly.json",
        "../tests/bothStages.json"
     ], [
        "Stage 1 only",
        "Stage 2 only",
        "BothStages"
    ]
)

makePRCurve([
    "../tests/house1stStageOnly.json",
    "../tests/houseBothStages.json",
     "../tests/houseBothStages2.json",
     "../tests/houseBothStages3.json",
     "../tests/houseBothStages4.json",
     "../tests/houseBothStages5.json",
     "../tests/houseBothStages6.json",
     "../tests/houseBothStages7.json"
    ], [
        "Stage 1 only",
        "Both stages",
        "Both stages 2",
        "Both stages 3",
        "Both stages 4",
        "Both stages 5",
        "Both stages 6",
        "Both stages 7"
    ]
)