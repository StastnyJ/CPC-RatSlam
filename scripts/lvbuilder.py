#!/usr/bin/python3

#https://ros-developer.com/2017/12/09/density-based-spatial-clustering-dbscan-with-python-code/

import rospy
import pcl
import numpy as np
from sensor_msgs.msg import PointCloud2
import scipy
from scipy.spatial import ConvexHull
from sklearn import cluster
from datetime import datetime
from math import inf
from std_msgs.msg import String
from typing import List
from statistics import median
from utils.pointSearch import PointSearch, DistanceMatrixSearch, KDTreeSearch
from utils.utils import  pc2msg_to_points, buildPC2Message, point_to_data, rgbToHex
from colored_point_cloud_rat_slam_ros.msg import LVDescription
import pyransac3d as pyrsc
from model.PointNet import loadModel
import os
import torch
from torch.autograd import Variable

class ClusterDescription:    
    def __init__(self, cluster: np.array, colorScaleFactor: float):
        self._rawPoints = cluster
        self._rawCoordinates = np.array([p[:3] for p in cluster])
        self._convexHull = ConvexHull(self._rawCoordinates, incremental=False)
        self.color = self._getAverageColor(colorScaleFactor)
        self.center, self.bondries = self._getBondryBox()
        self.convexHullVolume = self._convexHull.volume
        self.convexHullArea = self._convexHull.area
        self.clusterSize = len(cluster)
        self.shape, self.shapeConfidence = self._detectShape()
    
    def _getAverageColor(self,  colorScaleFactor: float):
        r = median(p[3] / colorScaleFactor for p in self._rawPoints)
        g = median(p[4] / colorScaleFactor for p in self._rawPoints)
        b = median(p[5] / colorScaleFactor for p in self._rawPoints)
        return (int(r), int(g), int(b))

    def _getBondryBox(self):
        xMin, xMax, yMin, yMax, zMin, zMax = (inf, -inf, inf, -inf, inf, -inf)
        for p in self._convexHull.vertices:
            point = self._rawCoordinates[p]
            xMin = min(xMin, point[0])
            xMax = max(xMax, point[0])
            yMin = min(yMin, point[1])
            yMax = max(yMax, point[1])
            zMin = min(zMin, point[2])
            zMax = max(zMax, point[2])
        return (((xMin + xMax)/2, (yMin + yMax) / 2, (zMin + zMax) / 2), (xMax - xMin, yMax - yMin, zMax - zMin))

    def _detectShape(self):
        bestConfidence = 0
        bestClass = "unknown"
        points = np.array([self._rawCoordinates[x] for x in self._convexHull.vertices])
        shapes = [("cylinder", pyrsc.Cylinder()), ("cuboid", pyrsc.Cuboid()), ("sphere", pyrsc.Sphere()), ("plane", pyrsc.Plane())]
        for shapeClass, shapeModel in shapes:
            inliners = shapeModel.fit(points, thresh=0.005, maxIteration = 10)[-1] # TODO improve
            confidence = len(inliners) / len(points)
            if confidence >= bestConfidence:
                bestConfidence = confidence
                bestClass = shapeClass
        return (bestClass, bestConfidence)

    def __str__(self):
        return '{"color":"' + rgbToHex(self.color) + \
                '","center":[' + ",".join([str(x) for x in self.center]) + \
                '],"bondrySize":[' + ",".join([str(x) for x in self.bondries])  +\
                '],"volume":' + str(self.convexHullVolume) + \
                ',"area":' +  str(self.convexHullArea) + \
                ',"shape":"' + self.shape + \
                '","shapeConfidence":' + str(self.shapeConfidence) + \
                ',"clusterSize":' + str(self.clusterSize) + '}'


class DBScan:
    def __init__(self, points: np.array, epsilon: float, minimumPoints: int):
        self.points = points
        self.epsilon = epsilon
        self.minimumPoints = minimumPoints
        self._result = None
        self._clusters = None
        self._unclassified = None
        self._search: PointSearch = DistanceMatrixSearch(points) 
        # self._search: PointSearch = KDTreeSearch(points) 
        
    def exec(self):
        self.reset()      
        m, n = self.points.shape
        visited = np.zeros(m, 'int')
        clusters = np.zeros(m)
        clusterIndex = 1
        neighbors = []
        for i in range(m):
            if visited[i] == 0:
                visited[i] = 1
                neighbors = self._search.findNeighbors(i, self.epsilon)
                if len(neighbors) >= self.minimumPoints:
                    clusters[i] = clusterIndex
                    neighbors = set(neighbors)
                    self._expandCluster(neighbors, visited, clusters, clusterIndex)
                    clusterIndex += 1
        self._result = clusters 
 
    def _expandCluster(self, pointNeighbors, visited, clusters, clusterIndex):
        neighbors=[]
        while len(pointNeighbors) > 0:
            i = pointNeighbors.pop()
            if visited[i] == 0:
                visited[i] = 1
                neighbors = self._search.findNeighbors(i, self.epsilon)
                if len(neighbors) >= self.minimumPoints:
                    for j in neighbors:
                        if j not in pointNeighbors:
                            pointNeighbors.add(j)
            if clusters[i] == 0:
                clusters[i] = clusterIndex 

    def getClustersMask(self):
        if self._result is None:
            self.exec()
        return self._result
    
    def getClusters(self):
        if self._result is None:
            self.exec()
        if self._clusters is None:
            self._clusters = [[] for _ in range(int(max(self._result)))]
            for (i, point) in enumerate(self.points):
                if self._result[i] > 0:
                    self._clusters[int(self._result[i]) - 1].append(point)
        return self._clusters

    def getUnclassifiedPoints(self):
        if self._result is None:
            self.exec()
        if self._unclassified is None:
            self._unclassified = []
            for (i, point) in enumerate(self.points):
                if self._result[i] <= 0:
                    self._unclassified.append(point)
        return self._unclassified

    def reset(self):
        self._result = None
        self._clusters = None
        self._unclassified = None


class Node:
    def __init__(self, subscribeTopic, topicOut, visualizationPublishTopic="", convexHullsVisualizationPublishTopic="", colorDimesionsScaling=1.0):
        self.publishViz = len(visualizationPublishTopic) > 0
        self.publisHullsViz = len(convexHullsVisualizationPublishTopic) > 0
        if self.publishViz:
            self.vizPublisher = rospy.Publisher(visualizationPublishTopic, PointCloud2 , queue_size=1)
            self.vizColors = [
                    (255,0,0),(0,0,255),(0,255,0),(127,0,255),(255,0,127),(0,255,255),(0,102,0),(0,0,153),
                    (255,255,0),(153,0,76),(102,51,0),(160,160,160),(255,255,255),(0,0,0),(153,255,255)
                ]
        if self.publisHullsViz:
            self.hullVizPublisher = rospy.Publisher(convexHullsVisualizationPublishTopic, PointCloud2 , queue_size=1)

        self.publisher = rospy.Publisher(topicOut, LVDescription, queue_size=1)
        self.subscriber = rospy.Subscriber(subscribeTopic, PointCloud2, self._onReceive)
        self.colorDimesionsScaling = colorDimesionsScaling
        self.model = loadModel(os.path.dirname(os.path.realpath(__file__)) + "/model/cls_model_62.pth")

    def _onReceive(self, cloudMsg):
        points = pc2msg_to_points(cloudMsg, includeColor=True, ignoreFloor=True)
        points = np.multiply(points, np.array([np.array([
            1, 1, 1, self.colorDimesionsScaling, self.colorDimesionsScaling, self.colorDimesionsScaling
        ]) for _ in range(len(points))]))
        clusters = DBScan(points, 0.6, 10).getClusters()
        if self.publishViz:
            self.publishClustersVisualization(clusters, cloudMsg.header.frame_id)
        # start = datetime.now()
        descriptions = [ClusterDescription(c, self.colorDimesionsScaling) for c in clusters]
        # rospy.logwarn((datetime.now() - start).total_seconds())
        if self.publisHullsViz:
            self.publishConvexHullsVisualization(descriptions, cloudMsg.header.frame_id)
        msg = LVDescription()
        msg.data = "[" + ",".join([str(des) for des in descriptions]) + "]"
        msg.header.stamp = cloudMsg.header.stamp
        msg.features = self.extractFeatures(points)
        self.publisher.publish(msg)

    def extractFeatures(self, points):
        data = Variable(torch.tensor([[[float(x[0]) for x in points], [float(x[1]) for x in points], [float(x[2]) for x in points]]]))
        return self.model(data)[3].tolist()[0]
    
    def publishClustersVisualization(self, clusters, frameId):
        data = []
        for (i,cluster) in enumerate(clusters):
            for point in cluster:
                data += point_to_data(point[0:3], self.vizColors[i % len(self.vizColors)])
        self.vizPublisher.publish(buildPC2Message(data, frameId))

    def publishConvexHullsVisualization(self, clusterInformation: List[ClusterDescription], frameId: int):
        data = []
        for (i,info) in enumerate(clusterInformation):
            for p in info._convexHull.vertices:
                data += point_to_data(info._rawCoordinates[p], info.color)
        self.hullVizPublisher.publish(buildPC2Message(data, frameId))


if __name__ == '__main__':
    rospy.init_node('lv_builder')

    topicIn = rospy.get_param('~topic_in', 'rgb_cloud')
    topicOut = rospy.get_param('~topic_out', "current_scene_descripion")
    visualizationTopic = rospy.get_param('~topic_viz', "")
    convexHullsVisualizationTopic = rospy.get_param('~topic_viz_convex_hull', "")
    colorDimensionScale = rospy.get_param('~color_dimension_scale', 0.001)

    Node(
        topicIn, topicOut,
        visualizationPublishTopic=visualizationTopic,
        convexHullsVisualizationPublishTopic=convexHullsVisualizationTopic,
        colorDimesionsScaling=colorDimensionScale
    )
    while not rospy.is_shutdown():
        rospy.spin()
