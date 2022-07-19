import sensor_msgs.point_cloud2
import struct
import ctypes
import numpy as np
from sensor_msgs.msg import  PointCloud2, PointField
import rospy
from typing import Tuple

FLOOR_IGNORE_THRESHOLD = 0.1

def getColorFromPoint(point):
    colorCode = point[3] 
    s = struct.pack('>f' ,colorCode)
    i = struct.unpack('>l',s)[0]
    pack = ctypes.c_uint32(i).value
    r = (pack & 0x00FF0000)>> 16
    g = (pack & 0x0000FF00)>> 8
    b = (pack & 0x000000FF)
    return (r,g,b)

def pc2msg_to_points(msg, includeColor=True, ignoreFloor=False):
    points = []
    for point in sensor_msgs.point_cloud2.read_points_list(msg, skip_nans=True):
        x = point[0]
        y = point[1]
        z = point[2]
        if ignoreFloor and z <= FLOOR_IGNORE_THRESHOLD:
            continue
        if includeColor:
            r,g,b = getColorFromPoint(point)
            points.append(np.array([x,y,z,r,g,b]))
        else:
            points.append(np.array([x,y,z]))
    return np.array(points)

def point_to_data(point, rgb):
    data_segment = []
    for value in point:
        binTest = binary(value)
        bin1 = binTest[ 0: 8]
        bin2 = binTest[ 8:16]
        bin3 = binTest[16:24]
        bin4 = binTest[24:32]
        converted_value = [int(bin4,2),int(bin3,2),int(bin2,2),int(bin1,2)]
        data_segment = data_segment + converted_value
    data_segment = data_segment + [0, 0, 0, 0]   #paddig
    data_segment = data_segment + [rgb[0], rgb[1], rgb[2]]
    data_segment = data_segment + [0]       # Padding
    return data_segment

def binary(num):
    # return ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))
    return ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))


def rgbToHex(color: Tuple[int,int,int]) -> str:
    r,g,b = color
    return '0x' + hex(b + 256 * g + 256 * 256 * r)[2:].zfill(6)


def buildPC2Message(data, frameId: int):
    msg = PointCloud2()
    
    f1 = PointField()
    f2 = PointField()
    f3 = PointField()
    f4 = PointField()
    
    #msg.header.frame_id = "usb_cam"
    
    msg.height = 1
    #msg.width = 3
    msg.point_step = 20
    #msg.row_step = 30
    
    f1.name = "x"
    f1.offset = 0
    f1.datatype = 7
    f1.count = 1
    
    f2.name = "y"
    f2.offset = 4
    f2.datatype = 7
    f2.count = 1
    
    f3.name = "z"
    f3.offset = 8
    f3.datatype = 7
    f3.count = 1
     
    f4.name = "rgb"
    f4.offset = 16
    f4.datatype = 7
    f4.count = 1
    
    msg.fields = [f1, f2, f3, f4]
    msg.is_dense = False
    
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frameId
    msg.data = data
    msg.width = int(len(data)/msg.point_step)

    return msg