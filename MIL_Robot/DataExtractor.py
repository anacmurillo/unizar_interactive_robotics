import roslib
import rospy
import sys
import struct
import rosbag
import os
import timeit
import skimage
import cv2
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from Scene.Scene import *

class DataExtractor():
    def __init__(self,Label,end):
        self.path_base = "tmp/"
        self.label=Label
        self.init = []
        self.end = int(end)
        self.rgb_top = []
        self.depth_top = []
        self.depth_front = []
        self.bridge = CvBridge()
        self.depth_top_sub = rospy.Subscriber("/k1/depth_registered/hw_registered/image_rect", Image, self.depth_top_callback)
        self.image_top_sub = rospy.Subscriber("/k1/rgb/image_rect_color", Image, self.rgb_front_callback)
        self.depth_front_sub = rospy.Subscriber("/k2/depth_registered/hw_registered/image_rect", Image, self.depth_front_callback)
        self.image_front_sub = rospy.Subscriber("/k2/rgb/image_rect_color", Image, self.rgb_front_callback)


    def start(self):
        self.init=[len(self.rgb_top),len(self.rgb_front),len(self.depth_top),len(self.depth_front)]

    def finish(self):

        self.image_top_sub.unregister()

        self.depth_top_sub.unregister()

        self.image_front_sub.unregister()

        self.depth_front_sub.unregister()

        self.rgb_top=self.rgb_top[self.init[0]:]
        self.rgb_front=self.rgb_front[self.init[1]:]
        self.depth=self.depth[self.init[2]:]
        self.depth2=self.depth2[self.init[3]:]

        i = 0
        for img in self.rgb_top:
            cv2.imwrite("tmp/rgb_top_"+i.__str__()+".jpg",img)
            i+=1
        i=0

        for img in self.rgb_front:
            cv2.imwrite("tmp/rgb_front_"+i.__str__()+".jpg",img)
            i+=1
        i=0
        for img in self.depth_top:
            np.save("tmp/depth_top_"+i.__str__()+".npy",img)
            i+=1
        i=0

        for img in self.depth_front:
            np.save("tmp/depth_front_"+i.__str__()+".npy",img)
            i+=1
        data =[]
        minimum = min(len(self.rgb_top),len(self.rgb_front),len(self.depth_top),len(self.depth_front))
        for i in xrange(minimum):
            files = ["tmp/rgb_front_"+i.__str__()+".jpg",
                         "tmp/depth_front_" + i.__str__() + ".npy",
                         "front_mask.png",
                         "tmp/rgb_top_"+i.__str__()+".jpg",
                         "tmp/depth_top_" + i.__str__() + ".npy",
                         "top_mask.png"]
            Scen = Scene(files[0], files[1], files[2], files[3], files[4], files[5], files)
            data.append(Scen)
            if self.end !=0 and i >= self.end:
                break
        return data


    def rgb_front_callback(self, ros_image):
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_image, "passthrough")
            self.rgb_front.append(frame)
        except CvBridgeError, e:
            print e
        # Convert the image to a Numpy array since most cv2 functions
        # require Numpy arrays.

    def rgb_top_callback(self, ros_image):
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_image, "passthrough")
            self.rgb_top.append(frame)
        except CvBridgeError, e:
            print e

    def depth_top_callback(self, ros_image):
        try:
            # The depth image is a single-channel float32 image
            depth_image = self.bridge.imgmsg_to_cv2(ros_image, "passthrough")
            depth_array = np.array(depth_image, dtype=np.float32)
            depth_array = np.nan_to_num(depth_array)
            depth_array = depth_array * 1000
            self.depth_top.append(depth_array)
        except CvBridgeError, e:
            print e
            # Convert the depth image to a Numpy array since most cv2 functions
            # require Numpy arrays.

    def depth_front_callback(self, ros_image):
        try:
            # The depth image is a single-channel float32 image
            depth_image = self.bridge.imgmsg_to_cv2(ros_image, "passthrough")
            depth_array = np.array(depth_image, dtype=np.float32)
            depth_array = np.nan_to_num(depth_array)
            depth_array = depth_array * 1000
            self.depth_front.append(depth_array)
        except CvBridgeError, e:
            print e
