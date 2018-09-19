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

class cvBridgeDemo():
    def __init__(self,Label,path,init,end):
        def nothing(x):
            pass
        self.path_base = "tmp/"
        self.label=Label
        self.init = int(init)
        self.end = int(end)
        self.intR = 0000
        self.rgb = []
        self.depth = []
        self.depth2 = []
        self.trgb = []
        self.tdep = []
        self.usbrgb = []
        self.tusbrgb = []
        self.bridge = CvBridge()
        rospy.init_node('talker', anonymous=True)

        # self.RGB_sub = message_filters.Subscriber('/camera/rgb/image_rect_color', Image)
        # self.Depth_sub = message_filters.Subscriber('/camera/depth_registered/image_rect', Image)
        # rospy.Subscriber('Scene',String,callbackS)
        # self.clock_sub = rospy.Subscriber("/clock",Clock,self.clock_callback)
        # self.ts = message_filters.ApproximateTimeSynchronizer([self.RGB_sub, self.Depth_sub], 1000,0.02)
        self.image_sub = rospy.Subscriber("/k1/depth_registered/hw_registered/image_rect", Image, self.depth_callback)
        self.depth_sub = rospy.Subscriber("/k1/rgb/image_rect_color", Image, self.usb1_callback)
        self.image2_sub = rospy.Subscriber("/k2/depth_registered/hw_registered/image_rect", Image, self.depth2_callback)
        self.depth2_sub = rospy.Subscriber("/k2/rgb/image_rect_color", Image, self.usb2_callback)
        # self.ts.registerCallback(self.image_callback)

    def finish(self):

        self.image_sub.unregister()

        self.depth_sub.unregister()

        self.image2_sub.unregister()

        self.depth2_sub.unregister()
        i = 0

        for img in self.rgb:
            cv2.imwrite("/home/pazagra/MIL/tmp/rgb_"+i.__str__()+".jpg",img)
            i+=1
        i=0

        for img in self.usbrgb:
            cv2.imwrite("/home/pazagra/MIL/tmp/rgb2_"+i.__str__()+".jpg",img)
            i+=1
        i=0
        for img in self.depth:
            np.save("/home/pazagra/MIL/tmp/depth_"+i.__str__()+".npy",img)
            i+=1
        i=0

        for img in self.depth2:
            np.save("/home/pazagra/MIL/tmp/depth2_"+i.__str__()+".npy",img)
            i+=1
        i=0
        for img in self.depth2:
            cv2.imwrite("/home/pazagra/MIL/tmp/mask_" + i.__str__() + ".jpg",np.zeros((480,640)))
            i+=1
        i=0

        for img in self.depth2:
            cv2.imwrite("/home/pazagra/MIL/tmp/mask2_" + i.__str__() + ".jpg",np.zeros((480,640)))
            i+=1
        i=0
        data =[]
        for i in xrange(len(self.rgb)):
            files = ["/home/pazagra/MIL/tmp/rgb2_"+i.__str__()+".jpg",
                         "/home/pazagra/MIL/tmp/depth2_" + i.__str__() + ".npy",
                         "/home/pazagra/MIL/tmp/mask2_" + i.__str__() + ".jpg",
                         "/home/pazagra/MIL/tmp/rgb_"+i.__str__()+".jpg",
                         "/home/pazagra/MIL/tmp/depth_" + i.__str__() + ".npy",
                         "/home/pazagra/MIL/tmp/mask_" + i.__str__() + ".jpg"]
            Scen = Scene(files[0], files[1], files[2], files[3], files[4], files[5], files)
            data.append(Scen)
            if self.end !=0 and i >= self.end:
                break
        return data


    def usb2_callback(self, ros_image):
        # bridge = CvBridge()
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_image, "passthrough")
            self.usbrgb.append(frame)
        except CvBridgeError, e:
            print e
        # Convert the image to a Numpy array since most cv2 functions
        # require Numpy arrays.
        # frame = np.array((220,200,2), dtype=np.uint8)
        # self.tusbrgb.append(t)

    def usb1_callback(self, ros_image):
        # bridge = CvBridge()
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_image, "passthrough")
            self.rgb.append(frame)
        except CvBridgeError, e:
            print e
        # Convert the image to a Numpy array since most cv2 functions
        # require Numpy arrays.
        # frame = np.array((220,200,2), dtype=np.uint8)

        # self.trgb.append(t)

    def depth_callback(self, ros_image):
        try:
            # The depth image is a single-channel float32 image
            depth_image = self.bridge.imgmsg_to_cv2(ros_image, "passthrough")
            depth_array = np.array(depth_image, dtype=np.float32)
            depth_array = np.nan_to_num(depth_array)
            depth_array = depth_array * 1000
            self.depth.append(depth_array)
        except CvBridgeError, e:
            print e
            # Convert the depth image to a Numpy array since most cv2 functions
            # require Numpy arrays.
        # self.tdep.append(t)

    def depth2_callback(self, ros_image):
        try:
            # The depth image is a single-channel float32 image
            depth_image = self.bridge.imgmsg_to_cv2(ros_image, "passthrough")
            depth_array = np.array(depth_image, dtype=np.float32)
            depth_array = np.nan_to_num(depth_array)
            depth_array = depth_array * 1000
            self.depth2.append(depth_array)
        except CvBridgeError, e:
            print e
            # Convert the depth image to a Numpy array since most cv2 functions
            # require Numpy arrays.
        # self.tdep.append(t)