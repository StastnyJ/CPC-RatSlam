#!/usr/bin/python3

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2
from ratslam_ros.msg import ViewTemplate
from typing import List, Tuple
from math import inf, exp, sqrt
import json
# from utils.colorUtils import getColorDifference
import numpy as np
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from datetime import datetime
from colored_point_cloud_rat_slam_ros.msg import LVDescription
import os
import torch
import torch.nn as nn


def loadModel(file: str, modelNumber: int):
    model = None
    if modelNumber == 1:
        model = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 1), nn.Sigmoid())
    if modelNumber == 2:
        model = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 64), nn.Linear(64, 1), nn.Sigmoid())
    if modelNumber == 3:
        model = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 64), nn.Linear(64, 1), nn.Sigmoid())
    if modelNumber == 4:
        model = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 64), nn.Linear(64, 32), nn.Linear(32, 1), nn.Sigmoid())
    model.load_state_dict(torch.load(file))
    model.eval()
    return model



def getColorDifference(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    # color1Rgb = sRGBColor(c1[0] / 255.0, c1[1] / 255.0, c1[2] / 255.0)
    # color2Rgb = sRGBColor(c2[0] / 255.0, c2[1] / 255.0, c2[2] / 255.0)
    # color1Lab = convert_color(color1Rgb, LabColor)
    # color2Lab = convert_color(color2Rgb, LabColor)
    # de = delta_e_cie2000(color1Lab, color2Lab)
    # return de
    return sqrt((c1[0] - c2[0])**2 + (c1[0] - c2[0])**2 + (c1[0] - c2[0])**2)

class Params:
    def __init__(self, raw: List[float]):
        self.colorThreshold = raw[0]
        self.colorA = raw[1]
        self.colorWeight = raw[2]

        self.distanceThreshold = raw[3]
        self.distanceA = raw[4]
        self.distanceWeight = raw[5]

        self.sizeThreshold = raw[6]
        self.sizeA = raw[7]
        self.sizeWeight = raw[8]

        self.volumeAreaRatioThreshold = raw[9]
        self.volumeAreaRatioA = raw[10]
        self.volumeAreaRatioWeight = raw[11]


def sig(x:float, x0: float, a:float) -> float:
    return 1.0 / (1.0 + exp(-a * (x - x0)))

def hexToRgb(hex: str) -> Tuple[int, int, int]:
    if len(hex[2:]) < 6:
        hex = "0x0" + hex[2:]
    return (int(hex[2:4], 16), int(hex[4:6], 16), int(hex[6:8], 16))


class LVObject:
    def __init__(self,
        color: Tuple[int,int,int], center: Tuple[float,float,float], bondrySize: Tuple[float, float, float], 
        volume: float, area: float, shape: str, clusterSize: int, params:Params
    ):
        self.color = color
        self.center = center
        self.bondrySize = bondrySize
        self.volume = volume
        self.area = area
        self.clusterSize = clusterSize
        self.params = params

    def findMostSimilarObject(self, others: List["LVObject"]) -> Tuple[float, "LVObject"]:
        bestSimilarity = 0.0
        bestObj = None
        for o in others:
            actSim = self.compareObjects(o)
            if actSim > bestSimilarity:
                bestSimilarity = actSim
                bestObj = o
        return (bestSimilarity, bestObj)


    def compareObjects(self, other: "LVObject") -> float:
        colorSim = 1 - sig(getColorDifference(self.color, other.color), self.params.colorThreshold, self.params.colorA)
        distanceSim = 1 - sig(np.linalg.norm(np.subtract(self.center, other.center)), self.params.distanceThreshold, self.params.distanceA)
        sizeSim = 1 - sig(abs(self.volume - other.volume), self.params.sizeThreshold, self.params.sizeA)
        volumeAreaRatioSim = 1 - sig(abs(self.volume / self.area - other.volume / other.area), self.params.volumeAreaRatioThreshold, self.params.volumeAreaRatioA)
        weightsSum = self.params.colorWeight + self.params.distanceWeight + self.params.sizeWeight + self.params.volumeAreaRatioWeight
        return (colorSim * self.params.colorWeight + distanceSim * self.params.distanceWeight + sizeSim * self.params.sizeWeight + volumeAreaRatioSim * self.params.volumeAreaRatioWeight) / weightsSum


class LV:
    def __init__(self, objects: List[LVObject], params: Params, features: List[float]):
        self.objects = objects
        self.params = params
        self.features = features

    @staticmethod
    def parseLV(raw: str, params: Params, features: List[float]) -> "LV":
        data = json.loads(raw)
        return LV([LVObject(hexToRgb(d["color"]), d["center"], d["bondrySize"], d["volume"], d["area"], "shape", d["clusterSize"], params) for d in data], params, features)

    def match(self, other: "LV", createThreshold: float, rejectThreshold: float, nn) -> float:
        start = datetime.now() # TIME MEASURE
        particleCount = 0
        res = 0.0
        if len(self.objects) == 0:
            return 1.0 if len(other.objects) == 0 else 0.0
        for o in self.objects:
            res += o.clusterSize * o.findMostSimilarObject(other.objects)[0]
            particleCount += o.clusterSize
        result = res / particleCount
        if nn is not None:
            if result >= createThreshold:
                nnResult = nn(torch.tensor(self.features + other.features))
                if nnResult >= rejectThreshold:
                    return result
                else:
                    return 0.0
        time = (datetime.now() - start).total_seconds() # END TIME MEASURE

        with open("matchingTimeMeasure.txt", "a") as f:
            f.write(str(time) + "\n")

        return result
        


class Node:
    def __init__(
        self, topicIn: str, topicOut: str, 
        threshold = 1.0, s1Threshold = 1.0, s2Threshold = 1.0, 
        useSecondStage = True, paramsArray: List[float] = []
    ):
        self.publisher = rospy.Publisher(topicOut, ViewTemplate, queue_size=1)
        self.subscriber = rospy.Subscriber(topicIn, LVDescription, self._onReceive)
        self.threshold = threshold
        self.s1Threshold = s1Threshold
        self.s2Threshold = s2Threshold
        self.useSecondStage = useSecondStage
        self._savedViews: List[LV] = []
        self._params = Params(paramsArray)
        if useSecondStage:
            self._nn = loadModel(os.path.dirname(os.path.realpath(__file__)) + "/model/model1.pth", 1)
        else:
            self._nn = None

    def _onReceive(self, rawData: LVDescription):
        currentView = LV.parseLV(rawData.data, self._params, rawData.features)
        start = datetime.now()
        bestViewIndex = -1
        bestSimilarity = -inf
        for (i, otherView) in enumerate(self._savedViews):
            similarity = currentView.match(otherView, self.s1Threshold, self.s2Threshold, self._nn) # TODO var
            # similarity = currentView.match(otherView, 0.646, 0.4, self._nn) # TODO var
            # similarity = currentView.match(otherView, 0.84, 0.00134, self._nn) # TODO var
            if similarity > bestSimilarity:
                bestViewIndex = i
                bestSimilarity = similarity
        rospy.logwarn(bestSimilarity)
        if bestSimilarity < self.threshold:
            bestViewIndex = len(self._savedViews)
            self._savedViews.append(currentView)
        msg = ViewTemplate()
        msg.current_id = bestViewIndex
        msg.relative_rad = bestSimilarity
        msg.header.stamp = rawData.header.stamp
        self.publisher.publish(msg)


if __name__ == '__main__':
    rospy.init_node('lv')

    topicIn = rospy.get_param('~topic_in', 'current_scene_descripion')
    topicOut = rospy.get_param('~topic_out', "irat_red/LocalView/Template")
    threshold = rospy.get_param('~new_view_threshold', 1.0)
    s1Threshold = rospy.get_param('~s1_threshold', 1.0)
    s2Threshold = rospy.get_param('~s2_threshold', 1.0)
    paramsArray = json.loads(rospy.get_param("~params_array", "[1.58773, 22.3013, 0.0406, 0.221766, 4.691886, 0.82588, 0.026518, 15.95429, 0.058236, 11.919109, 7.489252, 0.79746627]"))
    useSecondStage = rospy.get_param("~use_second_stage", True)

    Node(
        topicIn, topicOut,
        threshold, s1Threshold, s2Threshold,
        useSecondStage, paramsArray
    )


    while not rospy.is_shutdown():
        rospy.spin()

#        1.3412207,11.51117396,0.07464632,
#        0.04870507,8.427029,0.93453884,
#        0.72795924,6.07578882,0.09846648,
#        6.37736941,4.5353558,0.27635345

#threshold=0.737







#  threshold=3.80785060e-01,
#  paramsArray=[6.89214682e+00, 1.80208393e+01, 1.49458688e-01, 6.44444083e-02, 1.74226278e+01, 9.28374238e-01, 5.41219509e-01, 2.66566342e+00, 1.33238250e-02, 1.72236808e+01, 1.47960297e+01, 2.48803591e-01]
