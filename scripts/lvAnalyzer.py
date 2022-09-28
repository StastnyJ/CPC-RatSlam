#!/usr/bin/python3

import rospy
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2
from ratslam_ros.msg import ViewTemplate
from std_msgs.msg import Time
from typing import List, Tuple, Dict
from math import inf, asin, atan2, pi
from std_msgs.msg import String
import matplotlib.pyplot as plt
import os

import cv2
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()

imagesQueueSize = 100


def putMultilineText(img, lines, bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType):
    y0 = bottomLeftCornerOfText[1]
    dy = int(2.5 * cv2.getTextSize(lines[0], font, fontScale, thickness)[1])
    for i, l in enumerate(lines):
        cv2.putText(img, l, (bottomLeftCornerOfText[0], y0 + i*dy), font, fontScale, fontColor, thickness, lineType)

def eulerFromQuaternion(x, y, z, w):
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = atan2(t0, t1)
    
    t2 = 2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch= asin(t2)
    
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = atan2(t3, t4)
    
    return (roll / pi + 2) % 2, (pitch / pi + 2) % 2, (yaw / pi + 2) % 2

class Analyzer:
    def __init__(self, storeImages: bool = True, fpPosThreshold: float = 0.8, fpRotThreshold: float = 0.4, fnPosThreshold: float = 0.2, fnRotThreshold: float = 0.04, complexAnal:bool = False):
        self.storeImages = storeImages
        self.all: Dict[int, Tuple[Tuple[float, float, float], CompressedImage]] = {}

        self.falsePositives = 0
        self.falseNegatives = 0
        self.totalReceived = 0

        self.fpPosThreshold = fpPosThreshold
        self.fpRotThreshold = fpRotThreshold
        self.fnPosThreshold = fnPosThreshold
        self.fnRotThreshold = fnRotThreshold

        self.complexAnal = complexAnal
        if complexAnal:
            self.falsePositiveDetails = []
            self.falseNegativeDetails = []


    def insert(self, id: int, pos: Tuple[float, float, float], image: CompressedImage = None) -> str:
        self.totalReceived += 1

        isFP = self.isFP(id, pos)
        isFN = self.isFN(id, pos)
        isNew = False

        if self.complexAnal and isFP:
            self.findFPDetails(id, pos)

        if id not in self.all:
            self.all[id] = (pos, image)
            isNew = True

        if isFP:
            self.falsePositives += 1      
        
        if isFN:
            self.falseNegatives += 1

        return "FP" if isFP else "FN" if isFN else "TN" if isNew else "TP" 

    def isFP(self, currentId: int, currentPos: Tuple[float, float, float]) -> bool:
        if currentId not in self.all:
            return False
        x,y,a = currentPos
        mx, my, ma = self.all[currentId][0]
        return (mx - x)**2 + (my - y) ** 2 > self.fpPosThreshold ** 2 or min(abs(a - ma), abs(abs(a - ma) - 2)) > self.fpRotThreshold

    def isFN(self, currentId: int, currentPos: Tuple[float, float, float]) -> bool:
        if currentId in self.all:
            return False
        x,y,a = currentPos
        for (mx, my, ma),_ in self.all.values():
            if (mx - x)**2 + (my - y) ** 2 <= self.fnPosThreshold ** 2 and  min(abs(a - ma), abs(abs(a - ma) - 2))  <= self.fnRotThreshold:
                return True
        return False

    def findFPDetails(self, currentId: int, currentPos: Tuple[float, float, float]):
        x,y,a = currentPos
        mx, my, ma = self.all[currentId][0]
        dist = (mx - x)**2 + (my - y) ** 2
        orientation = min(abs(a - ma), abs(abs(a - ma) - 2)) 
        bestDist = dist
        bestOrientation = orientation
        bestImg = None
        for (mx, my, ma), img in self.all.values():
            d = (mx - x)**2 + (my - y) ** 2
            o =  min(abs(a - ma), abs(abs(a - ma) - 2)) 
            if d < bestDist:
                bestDist = d
                if self.storeImages:
                    bestImg = img
            if o < bestOrientation:
                bestOrientation = o
        self.falsePositiveDetails.append((dist, orientation, bestDist, bestOrientation, bestImg))
        self.saveDetails()

    def getStoredLVs(self):
        return len(self.all.keys())

    def getMatchedImage(self, id):
        return self.all[id][1]

    def __str__(self):
        return "saved: " + str(len(self.all.keys())) + ", FP: " + str(self.falsePositives) + ", FN:"  + str(self.falseNegatives) + ", total: " + str(self.totalReceived) + ", accuracy: " + str((self.totalReceived - self.falseNegatives - self.falsePositives) / self.totalReceived)

    def saveDetails(self):
        with open("/home/stastnyj/Dev/ros/CPC-RatSlam/src/colored_point_cloud_rat_slam_ros/anal/fpDetails.txt", "w") as f:
            f.write("\n".join([";".join([str(x) for x in row[:-1]]) for row in self.falsePositiveDetails]))

class Node:
    def __init__(self, lvTopic: str, cameraTopic: str, odometryTopic: str, fpPosThreshold: float, fpRotThreshold: float):
        self.lvSubscriber = rospy.Subscriber(lvTopic, ViewTemplate, self._onLvReceive)
        self.cameraSubscriber = rospy.Subscriber(cameraTopic, CompressedImage, self._onImageReceive)
        self.odomSubscriber = rospy.Subscriber(odometryTopic, Odometry, self._onOdomMessageReceive)
        self.imagesQueue: List[CompressedImage] = []
        self.odomQueue: List[Odometry] = []
        self.analyzer = Analyzer(storeImages=True,fpPosThreshold=fpPosThreshold, fpRotThreshold=fpRotThreshold, complexAnal=True)

        self.sims = []

    def _onLvReceive(self, rawData: ViewTemplate):
        currentId = rawData.current_id
        imageNow = self.findBestImage(rawData.header.stamp)
        if imageNow is None:
            return
        odomNow = self.findBestOdom(rawData.header.stamp)

        pos = (
            odomNow.pose.pose.position.x, 
            odomNow.pose.pose.position.y,
            eulerFromQuaternion(
                odomNow.pose.pose.orientation.x,
                odomNow.pose.pose.orientation.y,
                odomNow.pose.pose.orientation.z,
                odomNow.pose.pose.orientation.w
            )[2]
        )

        insertResult = self.analyzer.insert(currentId, pos, imageNow)

        self.sims.append((rawData.relative_rad, "P" if insertResult == "TP" or insertResult == "FN" else "N"))

        cvImageNow = bridge.compressed_imgmsg_to_cv2(imageNow, "bgr8")
        firstMatchImg = bridge.compressed_imgmsg_to_cv2(self.analyzer.getMatchedImage(currentId), "bgr8") 
        rospy.logwarn(str(self.analyzer))
        Node.showMatch(cvImageNow, firstMatchImg, odomNow)
        if insertResult == "FP":
            bestImg = bridge.compressed_imgmsg_to_cv2(self.analyzer.falsePositiveDetails[-1][-1], "bgr8") if self.analyzer.falsePositiveDetails[-1][-1] is not None else None
            Node.saveFalsePositive(cvImageNow, firstMatchImg, None, self.analyzer.falsePositives)

    def findBestImage(self, stamp: Time) -> CompressedImage:
        return Node.findBestMessage(self.imagesQueue, stamp)

    def findBestOdom(self, stamp: Time) -> Odometry:
        return Node.findBestMessage(self.odomQueue, stamp)


    @staticmethod
    def findBestMessage(messages, stamp: Time):
        bestMsg = None
        bestTimeDiff = inf
        for msg in messages:
            timeDiff = abs((msg.header.stamp.secs + msg.header.stamp.nsecs / 1000000000) - (stamp.secs + stamp.nsecs / 1000000000))
            if timeDiff < bestTimeDiff:
                bestMsg = msg
                bestTimeDiff = timeDiff
        return bestMsg

    @staticmethod
    def saveFalsePositive(img1, img2, bestImg, id):
        img = cv2.hconcat([bestImg, img1, img2]) if bestImg is not None else cv2.hconcat([img1, img2])
        cv2.imwrite("/home/stastnyj/Dev/ros/CPC-RatSlam/src/colored_point_cloud_rat_slam_ros/anal/FPs/" + str(id) + ".jpg", img)

    @staticmethod
    def showMatch(img1, img2, odom: Odometry):
        img = cv2.hconcat([img1, img2])

        font                   = cv2.FONT_HERSHEY_PLAIN
        bottomLeftCornerOfText = (10,440)
        fontScale              = 0.8
        fontColor              = (0,0,0)
        thickness              = 1
        lineType               = 2

        putMultilineText(img, [
            'x: %.3f, y: %.3f, yaw: %.4f pi' % (
                odom.pose.pose.position.x, 
                odom.pose.pose.position.y, 
                eulerFromQuaternion(
                    odom.pose.pose.orientation.x,
                    odom.pose.pose.orientation.y,
                    odom.pose.pose.orientation.z,
                    odom.pose.pose.orientation.w
                )[2]
            ), 
        ], bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

        cv2.imshow("Image Window", img)
        cv2.waitKey(1)

    def _onImageReceive(self, msg: CompressedImage):
        self.imagesQueue.append(msg)
        while len(self.imagesQueue) > imagesQueueSize:
            self.imagesQueue.pop(0)

    def _onOdomMessageReceive(self, msg: Odometry):
        self.odomQueue.append(msg)
        while len(self.odomQueue) > imagesQueueSize:
            self.odomQueue.pop(0)

if __name__ == '__main__':
    rospy.init_node('lvAnal')

    lvTopic = rospy.get_param('~lv_topic', '/LocalView/Template')
    cameraTopic = rospy.get_param('~camera_topic', "camera/image/compressed")
    odometryTopic = rospy.get_param('~odometry_topic', "odom") 
    fpPosThreshold = rospy.get_param('~fpPosThreshold', 0.8) 
    fpRotThreshold = rospy.get_param('~fpRotThreshold', 0.4) 

    Node(lvTopic, cameraTopic, odometryTopic, fpPosThreshold, fpRotThreshold)
    while not rospy.is_shutdown():
        rospy.spin()
