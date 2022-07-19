#!/usr/bin/python3

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Time
from typing import List, Tuple, Dict
from colored_point_cloud_rat_slam_ros.msg import LVDescription
import json
from math import inf, asin, atan2, pi

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

class Node:
    def __init__(self, lvTopic: str, odometryTopic: str, outputFileName: str):
        self.lvSubscriber = rospy.Subscriber(lvTopic, LVDescription, self._onLvReceive)
        self.odomSubscriber = rospy.Subscriber(odometryTopic, Odometry, self._onOdomMessageReceive)
        self.odomQueue = []
        self.outputFileName = outputFileName
        self.data = []



    def _onLvReceive(self, rawData: LVDescription):
        currentOdom = self.findBestOdom(rawData.header.stamp)
        x = currentOdom.pose.pose.position.x
        y = currentOdom.pose.pose.position.y 
        a = eulerFromQuaternion(
            currentOdom.pose.pose.orientation.x,
            currentOdom.pose.pose.orientation.y,
            currentOdom.pose.pose.orientation.z,
            currentOdom.pose.pose.orientation.w
        )[2]
        self.data.append({"scene": rawData.data, "features": rawData.features, "position": (x, y, a)})
        with open(outputFile, "w") as f:
            f.write(json.dumps(self.data))



    def findBestOdom(self, stamp: Time) -> Odometry:
        return Node.findBestMessage(self.odomQueue, stamp)

    def _onOdomMessageReceive(self, msg: Odometry):
        self.odomQueue.append(msg)
        while len(self.odomQueue) > 100:
            self.odomQueue.pop(0)

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



if __name__ == '__main__':
    rospy.init_node('lvDatasetCreator')

    sceneDescriptionTopic = rospy.get_param('~sceneDescriptionTopic', 'current_scene_descripion')
    odometryTopic = rospy.get_param('~odometryTopic', "odom")
    outputFile = rospy.get_param("~outputFile", "dataset.json")

    Node(sceneDescriptionTopic, odometryTopic, outputFile)
    while not rospy.is_shutdown():
        rospy.spin()