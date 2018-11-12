import math

import cv2
import numpy as np

import Rect


class Point_Seg:

    def __init__(self):
        self.distance = 80
        self.var = (25.0 * math.pi) / 180.0
        self.base_angle = -0.5*math.pi
        self.candidates = []

    def add_candidates(self, Candidates):
        self.candidates = Candidates

    def inside(self,T, pt):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        b1 = sign(pt, T[0], T[1]) < 0.0
        b2 = sign(pt, T[1], T[2]) < 0.0
        b3 = sign(pt, T[2], T[0]) < 0.0
        return ((b1 == b2) and (b2 == b3))

    def PointatD2(self,P_ini,theta,d):
        x2 = P_ini[0] + d * np.sin(theta)
        y2 = P_ini[1] + d * np.cos(theta)
        if (y2 >480) or (x2>640):
            return None
        return (int(x2),int(y2))

    def PointatD(self,P_ini, theta, d):
        x2 = P_ini[0] + d * np.sin(theta)
        y2 = P_ini[1] + d * np.cos(theta)
        if (y2 >480) or (x2>640):
            return None
        return (int(x2), int(y2))

    def obtain_candidate(self,Point_ini,Angle,isTop,Image=None):
        #Set the initial position at a distance from the center of the hand
        L = []
        #Calculate the discrete points of the line
        for d in xrange(0,500,5):
            P = self.PointatD(Point_ini,math.radians(Angle),d)
            if P is None:
                break
            cv2.circle(Image,P,1,(255,255,0),1)
            #Search for the first candidate that contains part of the line
            for i in self.candidates:
                if i is None:
                    continue
                if isTop:
                    if Rect.contains(i.BB_top, P):
                        p1, p2 = i.BB_top.two_point()
                        cv2.rectangle(Image, p1, p2, (255,255,0), 2)
                        L.append(i)
                else:
                    if Rect.contains(i.BB_front, P):
                        L.append(i)
        cv2.imshow("Image",Image)
        cv2.waitKey(400)
        return L
