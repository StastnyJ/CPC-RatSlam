#!/usr/bin/python3

import rospy

from sensor_msgs.msg import PointCloud2, PointField, CompressedImage, CameraInfo
from std_msgs.msg import Time
import sensor_msgs.point_cloud2
import tf
from tf.transformations import euler_from_quaternion
from cv_bridge import CvBridge, CvBridgeError
from utils.utils import point_to_data, buildPC2Message
from math import inf
import cv2


import numpy as np

imgQueueSize = 100


def translation(vector, xyz):
    return list(vector[:3] + np.array(xyz)) + vector[3:]


def transformPoints(trans, quat, points):
    points = [qvMult(quat, point) for point in points]
    points = [translation(point, trans) for point in points]
    return points


def qvMult(q1, v1):
    v1_new = tf.transformations.unit_vector(v1[:3])
    q2 = list(v1_new)
    q2.append(0.0)
    unit_vector = tf.transformations.quaternion_multiply(
                    tf.transformations.quaternion_multiply(q1, q2),
                    tf.transformations.quaternion_conjugate(q1)
                    )[:3]

    vector_len = np.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    vector = unit_vector * vector_len
    return list(vector) + v1[3:]


def pc2msg_to_points(msg):
    points = []
    for point in sensor_msgs.point_cloud2.read_points(msg, skip_nans=True):
            x = point[0]
            y = point[1]
            z = point[2]
            points.append([x,y,z])
    return points

class Node:
    def __init__(self, pointCloudTopic: str, imageTopic: str, cameraInfoTopic: str, topicOut: str, goalFrame: str):
        self.publisher = rospy.Publisher(topicOut, PointCloud2, queue_size=1)
        self.pcSubscriber = rospy.Subscriber(pointCloudTopic, PointCloud2, self._onPCReceive)
        self.camInfoSubscriber = rospy.Subscriber(cameraInfoTopic, CameraInfo, self._onCamInfoReceive)
        self.imageSubscriber = rospy.Subscriber(imageTopic, CompressedImage, self._onImageReceive)
        self.baseFrame = goalFrame
        self.cameraMatrix: np.matrix = np.identity(3)

        self.transListener = tf.TransformListener()
        self.bridge = CvBridge()
        self.imagesQueue: List[CompressedImage] = []


    def findBestImage(self, stamp: Time):
        bestMsg = None
        bestTimeDiff = inf
        for msg in self.imagesQueue:
            timeDiff = abs((msg.header.stamp.secs + msg.header.stamp.nsecs / 1000000000) - (stamp.secs + stamp.nsecs / 1000000000))
            if timeDiff < bestTimeDiff:
                bestMsg = msg
                bestTimeDiff = timeDiff
        return bestMsg

    def _onPCReceive(self, msg: PointCloud2):
        imgMsg = self.findBestImage(msg.header.stamp)
        while not rospy.is_shutdown():
            try:
                (trans, quat) = self.transListener.lookupTransform(imgMsg.header.frame_id, msg.header.frame_id, rospy.Time(0))
                (transToBase, quatToBase) = self.transListener.lookupTransform(self.baseFrame, imgMsg.header.frame_id, rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            img = self.bridge.compressed_imgmsg_to_cv2(imgMsg, "bgr8")
            break
        points = pc2msg_to_points(msg)
        transformedPoints = np.array(transformPoints(trans, quat, points)).tolist()

        coloredPoints = []

        for x, y, z in transformedPoints:
            projected = cv2.projectPoints(np.array([x,y,z]), cv2.Rodrigues(np.identity(3))[0], cv2.Rodrigues(np.identity(3))[0], self.cameraMatrix, ())[0]
            col, row = (round(projected.item(0)), round(projected.item(1)))
            if 0 <= row and row < len(img) and 0 <= col and col < len(img[0]):
                rgb = img[row][col]
                coloredPoints.append([x,y,z] + list(rgb))

        data = []
        coloredPoints = transformPoints(transToBase, quatToBase, coloredPoints)
        for point in coloredPoints:
            data_segment = point_to_data(point[0:3], point[3:])
            data = data + data_segment

        msg = buildPC2Message(data, self.baseFrame)
        msg.row_step = msg.width * msg.point_step
        msg.header.stamp = imgMsg.header.stamp
        self.publisher.publish(msg)

    def _onImageReceive(self, msg: CompressedImage):
        self.imagesQueue.append(msg)
        while len(self.imagesQueue) > imgQueueSize:
            self.imagesQueue.pop(0)

    def _onCamInfoReceive(self, msg: CameraInfo):
        self.cameraMatrix = np.matrix(msg.K).reshape(3, 3)


if __name__ == '__main__':
    rospy.init_node('colorCloud')

    pointCloudTopic = rospy.get_param('~pc2_topic_in', 'velodyne_points')
    imageTopic = rospy.get_param('~image_topic_in', 'camera/rgb/image_raw')
    cameraInfoTopic = rospy.get_param('~camera_info_topic_in', 'camera/rgb/camera_info')
    topicOut = rospy.get_param('~pc2_topic_out', 'rgb_cloud')
    goalFrame = rospy.get_param('~goal_frame', 'odom')

    Node(pointCloudTopic, imageTopic, cameraInfoTopic, topicOut, goalFrame)

    while not rospy.is_shutdown():
        rospy.spin()
