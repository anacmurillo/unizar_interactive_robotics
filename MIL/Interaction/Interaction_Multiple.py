import math
import os
import time
import CNN
import cv2
import numpy as np
from Interaction import Interaction_Recogn,Masking,Evaluator

class Mutiple_Interaction:
    def __init__(self):
        self.Interaction = Interaction_Recogn.Interaction_Recogn(16,0.65,None)
        self.Interaction_r = Interaction_Recogn.Interaction_Recogn(16,0.65,None)
        self.Masquerade = Masking.Masking()
        self.interactions = ["Point","Show"]
        self.Evaluator = Evaluator.Evaluator(self.interactions)

    def Calculate_Interaction(self,FM,Labels = None):
        def new_angle(Scena):
            def PointatD(P_ini, theta, d):
                x2 = P_ini[0] + d * np.sin(theta)
                y2 = P_ini[1] + d * np.cos(theta)
                return (int(y2), int(x2))
            f_depth = np.load(Scena.Depth_front)
            f_depth *= 255.0 / f_depth.max()
            f_depth = np.array(f_depth, np.uint8)
            f_depth = np.dstack((f_depth, f_depth, f_depth))
            gray = cv2.cvtColor(f_depth, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(gray , 10, 200)
            kernel = np.ones((4, 4), np.uint8)
            edged = cv2.dilate(edged, kernel, 1)
            kernel = np.ones((3, 3), np.uint8)
            edged = cv2.erode(edged, kernel, 1)
            f_depth = np.dstack((edged,edged,edged))
            f_depth = np.array(f_depth,np.uint8)
            if "Hand_Pos" in Scena.Values.keys():
                f_f = f_depth[Scena.Values["Hand_Pos"][0] - 100:Scena.Values["Hand_Pos"][0] + 100,
                      Scena.Values["Hand_Pos"][1] - 100:Scena.Values["Hand_Pos"][1] + 100, :].copy()
                f_f[0:1, :, :] = 255
                f_f[-1:, :, :] = 255
                f_f[:, 0:1, :] = 255
                f_f[:, -1:, :] = 255
                l = []
                p_ini = (100, 100)
                while f_f[p_ini[0], p_ini[1], 0] != 0:
                    p_ini = (p_ini[0] + 1, p_ini[1] + 1)
                for i in xrange(20, 160):
                    d = 5
                    p2 = PointatD(p_ini, math.radians(i), d)
                    while f_f[p2[1], p2[0], 0] != 255:
                        d += 0.5
                        p2 = PointatD(p_ini, math.radians(i), d)
                    if d <= 200 and d > 4 and (p2[0] < 199 and p2[1] < 199 and p2[0] > 1 and p2[1] > 1):
                        cv2.line(f_f, p_ini, p2, (0, 255, 0), 1)
                        l.append((i, d, p2))
                if l == []:
                    return None
                l = sorted(l, key=lambda p: p[1],reverse=True)
                prt = l[0]
                return prt[0]

        def angle(Mask,Center):
            m = Mask[Center[0]-100:Center[0]+100,Center[1]-100:Center[1]+100]
            Moments = cv2.moments(m)
            if Moments["nu20"] - Moments["nu02"] != 0:
                art = math.atan2(2 * Moments["nu11"], Moments["nu20"] - Moments["nu02"])
                theta = 0.5 * art
                theta = abs(theta)
                return math.degrees(theta)
            return None
        video = cv2.VideoWriter('Outputs/Videos/'+FM.user+"_"+FM.action+'.avi',cv2.cv.CV_FOURCC('D', 'I', 'V', 'X'), 25.0, (640, 480))
        for Scena in FM.Images:
            dep_front = np.load(Scena.Depth_front).copy()
            # Masking Part
            mask,move = self.Masquerade.Mask(cv2.imread(Scena.RGB_front),cv2.imread(Scena.Mask_front), dep_front)
            #Out = self.Interaction.search_one_image(cv2.imread(Scena.RGB_front), np.load(Scena.Depth_front), mask)
            Out_right = self.Interaction.Class_One_Image(cv2.imread(Scena.RGB_front), np.load(Scena.Depth_front), mask,move,False, Scena.Skeleton)
            Out_left = self.Interaction_r.Class_One_Image(cv2.imread(Scena.RGB_front), np.load(Scena.Depth_front), mask,move,True, Scena.Skeleton)
            Scena.Skeleton = self.Interaction.prvs_skel
            Scena.Values["Canvas"] = self.Interaction.prvs_canvas
            if Out_left is None and Out_right is None:
                Out = None
            elif Out_left is None and Out_right is not None:
                Out = Out_right
                Scena.Values["Side"]="right"
            elif Out_left is not None and Out_right is None:
                Out = Out_left
                Scena.Values["Side"] = "left"
            else:
                if Out_left[2] > Out_right[2]:
                    Out = Out_left
                    Scena.Values["Side"] = "left"
                else:
                    Out = Out_right
                    Scena.Values["Side"] = "right"
            if Out is not None:
                Class, Center, p = Out
                if FM.Speech[0] == 'That':
                    Class = 'Point'
                Scena.Values["Mask"] = mask
                Scena.Values["Interaction_recognized"] = Class
                Scena.Values["Hand_Pos"] = Center
                Scena.Values["Interaction_posibility"] = p
                self.Evaluator.add_data(Class,p)
                I = Scena.Values["Canvas"].copy()
                cv2.putText(I,Class,(Center[1]-100,Center[0]-100),cv2.cv.CV_FONT_ITALIC,1,(255,0,0),2)
                cv2.circle(I,(Center[1],Center[0]),100,(0,0,255),3)
                cv2.circle(I, (Center[1], Center[0]), 2, (0, 0, 255), 2)
                video.write(I)
            else:
                video.write(cv2.imread(Scena.RGB_front))
        video.release()
        Result_n, Result_p, Class_p, Class_n, Total =self.Evaluator.calculate_output()
        FM.Values["Class_N"]=Class_n
        FM.Values["Result_N"]= Result_n
        FM.Values["Total_N"]= Total
        self.pvrs_angle = None
        for Scena in FM.Images:
            if "Hand_Pos" not in Scena.Values.keys():
                continue
            if Class_n == 'Point':
                Hand_angle = new_angle(Scena)
                if Hand_angle is None:
                    Hand_angle = angle(Scena.Values["Mask"], Scena.Values["Hand_Pos"])
            else:
                Hand_angle = angle(Scena.Values["Mask"], Scena.Values["Hand_Pos"])
                if Hand_angle is None:
                    Hand_angle = new_angle(Scena)
            # if self.pvrs_angle is None:
            #     self.pvrs_angle = Hand_angle
            # else:
            #     Hand_angle = (self.pvrs_angle + Hand_angle) / 2
            #     self.pvrs_angle = Hand_angle
            Scena.Values["Hand_Angle"] = Hand_angle
        print "En caso de usar el voto normal ha salido "+ Class_n + " con "+Result_n.__str__()+" de votos de un total de "+Total.__str__()+" votos."
        print "En caso de usar el voto porcentaje ha salido " + Class_p + " con un porcentaje medio de " + Result_p.__str__() + "."
