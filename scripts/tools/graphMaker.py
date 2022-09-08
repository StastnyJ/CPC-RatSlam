#!/usr/bin/python3

import plotly.express as px
import itertools
from math import sqrt
import json


def makeFpDetailsGraph(file, title):
    rawData = []

    with open(file, "r") as f:
        rawData = [tuple([float(x) for x in row[:-1].split(";")]) for row in f.readlines()]

    data = list(itertools.chain(*[
        [
            {"False positive number": i + 1, "distance": sqrt(d[0]), "category": "current"},
            # {"id": i + 1, "actual": sqrt(d[2]), "category": "best"}, 
        ][:1 if d[0] == d[2] else 2] for (i,d) in enumerate(rawData)
    ]))

    errors = [d["distance"] for d in data]

    print(title + ": " + str(max(errors)))

    fig = px.scatter(data, x="False positive number", y="distance", color="category", title=title, range_y=(0,3.5))
    fig.add_hline(y=0.8, annotation={"text": "threshold"})
    fig.add_hline(y=1.6, annotation={"text": "2x threshold"})
    fig.add_hline(y=2.4, annotation={"text": "3x threshold"})
    fig.add_hline(y=sum(errors)/len(errors), line_color="orange", annotation={"bgcolor": "orange", "text": "average = " + str(sum(errors) / len(errors)) + " m = " + str(sum(errors) / (len(errors) * 0.8) - 1) + " thresholds"} )
    fig.show()

def makeTimeCurve(files, curveNames, title):
    data = []
    averageTimes = []
    for (i, fn) in enumerate(files):
        rawData = []
        with open(fn, "r") as f:
            rawData = [float(x) * 1000 for x in f.readlines()]

        data += [{
            "approach": curveNames[i],
            "time (ms)": d,
            "scene number": index + 1
        } for (index, d) in enumerate(rawData)]

        averageTimes.append(sum(rawData) / len(rawData))

    fig = px.scatter(data, y="time (ms)",x="scene number", title=title, color="approach")
    for (i,t) in enumerate(averageTimes):
        fig.add_hline(y=t, line_color="orange", annotation={"bgcolor": "orange", "text": "average time [" + curveNames[i] + "]= " + str(t) + " ms"} )
    fig.show()

def makePRCurve(files, curveNames, title):
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
            "approach": curveNames[i]
        } for d in rawData]
    
    fig = px.line(data, y="precission",x="recall", title=title, color="approach", markers=True)
    fig.show()

    fig = px.line(data, y="accuracy",x="threshold", title=title, color="approach", markers=True)
    fig.show()


# makeTimeCurve(["../tests/warehouse/matchingTimes1stStage.txt", "../tests/warehouse/matchingTimesBoth.txt"], ["1st stage only", "Both stages"], "Warehouse LV matching times")
# makeTimeCurve(["../tests/warehouse/buildingTimes1stStage.txt", "../tests/warehouse/buildingTimesBoth.txt"], ["1st stage only", "Both stages"], "Warehouse LV building times")

# makeTimeCurve(["../tests/house/matchingTimes1stStage.txt", "../tests/house/matchingTimesBoth.txt"], ["1st stage only", "Both stages"], "House LV matching times")
# makeTimeCurve(["../tests/house/buildingTimes1stStage.txt", "../tests/house/buildingTimesBoth.txt"], ["1st stage only", "Both stages"], "House LV building times")

# makeTimeCurve(["../tests/hospital/matchingTimes1stStage.txt", "../tests/hospital/matchingTimesBoth.txt"], ["1st stage only", "Both stages"], "Hospital LV matching times")
# makeTimeCurve(["../tests/hospital/buildingTimes1stStage.txt", "../tests/hospital/buildingTimesBoth.txt"], ["1st stage only", "Both stages"], "Hospital LV building times")


makeFpDetailsGraph("../tests/warehouse/fpDetails1stOnly.txt", "Warehouse First stage only")
makeFpDetailsGraph("../tests/warehouse/fpDetailsBoth.txt", "Warehouse Both stages")
makeFpDetailsGraph("../tests/warehouse/fpDetailsRatSlam.txt", "Warehouse RatSlam")

makeFpDetailsGraph("../tests/house/fpDetailsFirst.txt", "House First stage only")
makeFpDetailsGraph("../tests/house/fpDetailsBoth.txt", "House Both stages")
makeFpDetailsGraph("../tests/house/fpDetailsRatSlam.txt", "House RatSlam")

makeFpDetailsGraph("../tests/hospital/fpDetails1stStage.txt", "Hospital First stage only")
makeFpDetailsGraph("../tests/hospital/fpDetailsBoth.txt", "Hospital Both stages")
makeFpDetailsGraph("../tests/hospital/fpDetailsRatSlam.txt", "Hospital RatSlam")


# makePRCurve([
#         "../tests/warehouse/1stStageOnly.json",
#         "../tests/warehouse/bothStages.json"
#      ], [
#         "Stage 1 only",
#         "BothStages"
#     ],
#     "Warehouse"
# )
# makePRCurve([
#         "../tests/house/1stStageOnly.json",
#         "../tests/house/bothStages.json"
#      ], [
#         "Stage 1 only",
#         "BothStages",
#     ],
#     "House"
# )


# makePRCurve([
#         "../tests/hospital/1stStageOnly.json",
#         "../tests/hospital/bothStages.json"
#      ], [
#         "Stage 1 only",
#         "BothStages"
#     ],
#     "Hospital"
# )