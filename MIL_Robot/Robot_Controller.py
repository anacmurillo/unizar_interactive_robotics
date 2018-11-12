import cv2
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from baxter_core_msgs.msg import EndpointState
import math
global p
p=[]

import rospy

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
import baxter_interface


def standar_calibration(robot = False):
    raw_input("For this calibration you need to put four small objects in the visible area of the robot and the two kinects. To calibrate the kinects you need to click in the center of the objects in the same order in both cameras. Press Enter when you are ready...")
    points =[]
    homo = None
    def p(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            points.append([x, y])
    rospy.init_node('talker', anonymous=True)
    bridge = CvBridge()
    if not robot:
        print "Calibration Two Kinects...."
        msg = rospy.wait_for_message("/k1/rgb/image_rect_color",Image)
        image = bridge.imgmsg_to_cv2(msg, "passthrough")
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',p)
        cv2.imshow('image',image)
        cv2.waitKey()
        msg = rospy.wait_for_message("/k2/rgb/image_rect_color",Image)
        image = bridge.imgmsg_to_cv2(msg, "passthrough")
        cv2.imshow('image',image)
        cv2.waitKey()
        pts_src = np.array(points[0:4])
        pts_dst = np.array(points[4:8])
        homo, status = cv2.findHomography(pts_src, pts_dst)
        np.save('Kinects.npy',homo)
    raw_input("For the calibration of the robot with the kinect use the same four objects. You must move the robot hand on top of the object and press enter. The first two should be the right hand and then the left hand. Press Enter when you are ready...")
    print "Calibration Kinect Robot"
    points2=[]
    msg = rospy.wait_for_message("/k1/rgb/image_rect_color",Image)
    image = bridge.imgmsg_to_cv2(msg, "passthrough")
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',p)
    cv2.imshow('image',image)
    cv2.waitKey()
    for p in points:
        p2 = do_point_transform([p[0],p[1],1.14])
        points2.append([p2[0],p2[1]])
    raw_input("Move the hand and press enter...")
    msg = rospy.wait_for_message('/robot/limb/right/endpoint_state', EndpointState)
    points2.append([msg.pose.position.x,msg.pose.position.y])
    raw_input("Move the hand and press enter...")
    msg = rospy.wait_for_message('/robot/limb/right/endpoint_state', EndpointState)
    points2.append([msg.pose.position.x,msg.pose.position.y])
    raw_input("Move the hand and press enter...")
    msg = rospy.wait_for_message('/robot/limb/left/endpoint_state', EndpointState)
    points2.append([msg.pose.position.x,msg.pose.position.y])
    raw_input("Move the hand and press enter...")
    msg = rospy.wait_for_message('/robot/limb/left/endpoint_state', EndpointState)
    points2.append([msg.pose.position.x,msg.pose.position.y])
    pts_src = np.array(points2[0:4])
    pts_dst = np.array(points2[4:8])
    homo2, status = cv2.findHomography(pts_src, pts_dst)
    np.save('KRobot.npy',homo2)
    return homo,homo2

def do_point_transform(data):
    [x, y,z] = data
    center_point = upper_left_to_zero_center(x, y, 640, 480)
    z = 1.1
    position = calc_geometric_location(center_point[0], center_point[1], z, 640, 480)
    return position

def upper_left_to_zero_center( x, y, width, height):
        '''Change referential from center to upper left'''
        return (x - int(width/2), y - int(height/2))

def calc_geometric_location( x_pixel, y_pixel, kinect_z, width, height):
        f = width / (2 * math.tan(math.radians(57/2))) #57 = fov angle in kinect spects
        d = kinect_z / f
        x = d * x_pixel
        y = d * y_pixel
        return [x,y,kinect_z]

def standard_move():
    #Moves the hands to a position that doesn't interfere with the kinects
    limb_left = baxter_interface.Limb('left')
    limb_right = baxter_interface.Limb('right')
    joint_angles = ik('left',0.709,0.51,0.15)
    if joint_angles != 0 and joint_angles != 1:
        limb_left.move_to_joint_positions(joint_angles)
    joint_angles = ik('right', 0.6,-0.57,0.22)
    if joint_angles != 0 and joint_angles != 1:
        limb_right.move_to_joint_positions(joint_angles)


def ik(limb,x,y,z=-0.05):
    ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest()
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    poses = {
        'left': PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point(
                    x=x,
                    y=y,
                    z=z,
                ),
                orientation=Quaternion(
                    x=0.00,#1,
                    y=1.0,#0.99,
                    z=0.0,#0.03,
                    w=0.0,#0.07,
                ),
            ),
        ),
        'right': PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point(
                    x=x,
                    y=y,
                    z=z,
                ),
                orientation=Quaternion(
                    x=0.0,#0.904,
                    y=1.0,# 0.422,
                    z=0.0,#0.056,
                    w=0.0,#-0.027,
                ),
            ),
        ),
    }

    ikreq.pose_stamp.append(poses[limb])
    try:
        rospy.wait_for_service(ns, 5.0)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException), e:
        rospy.logerr("Service call failed: %s" % (e,))
        return 1
    if (resp.isValid[0]):
        # print("SUCCESS - Valid Joint Solution Found:")
        # Format solution into Limb API-compatible dictionary
        limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
        # print limb_joints
        return limb_joints
    else:
        print("INVALID POSE - No Valid Joint Solution Found.")

    return 0


def move_hand(x,y):
    point_1 = do_point_transform([x,y,1.12])

    p1 = np.dot(h, np.array(point_1[0:2] + [1.0]).T)
    p1 = p1 / p1[2]
    if p1[1]<=0.0:
        limb_right = baxter_interface.Limb('right')
        joint_angles = ik('right', p1[0], p1[1])
        if joint_angles != 0 and joint_angles != 1:
            limb_right.move_to_joint_positions(joint_angles)
            cv2.waitKey(4000)
            standard_move()
            return 0
        else:
            return -1
    else:
        limb_left = baxter_interface.Limb('left')
        joint_angles = ik('left', p1[0], p1[1])
        if joint_angles != 0 and joint_angles != 1:
            limb_left.move_to_joint_positions(joint_angles)
            cv2.waitKey(4000)
            standard_move()
            return 0
        else:
            return -1