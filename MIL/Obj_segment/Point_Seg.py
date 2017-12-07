import math

import cv2
import numpy as np
from skimage.segmentation import slic

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

    def PointatD(self,P_ini,theta,d):
        x2 = P_ini[1] + d * np.sin(theta)
        y2 = P_ini[0] + d * np.cos(theta)
        if (y2 >640) or (x2>480):
            return None
        return (int(y2),int(x2))

    def obtain_candidate(self,Point_ini,Angle,isTop):
        #Set the initial position at a distance from the center of the hand
        Ini = self.PointatD(Point_ini, math.radians(Angle), 20)
        L = []
        #Calculate the discrete points of the line
        for d in xrange(20,100,5):
            P = self.PointatD(Ini,Angle,d)
            if P is None:
                continue

            #Search for the first candidate that contains part of the line
            for i in self.candidates:
                if i is None:
                    continue
                if isTop:
                    if Rect.contains(i.BB_top, P):
                        L.append(i)
                else:
                    if Rect.contains(i.BB_front, P):
                        L.append(i)
        return np.unique(L)
