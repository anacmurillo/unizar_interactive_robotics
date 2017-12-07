import math
import os
import time
import CNN
import cv2
import numpy as np
from Interaction import Interaction_Recogn,Masking,Evaluator

class Mutiple_Interaction:
    def __init__(self):
        self.Interaction = Interaction_Recogn.Interaction_Recogn(16,0.60,None)
        self.Masquerade = Masking.Masking()
        self.interactions = ["Point","Show"]
        self.Evaluator = Evaluator.Evaluator(self.interactions)

    def Calculate_Interaction(self,FM,Labels = None):

        def angle(Mask,Center):
            m = Mask[Center[0]-50:Center[0]+50,Center[1]-50:Center[1]+50]
            Moments = cv2.moments(m[:, :])
            if Moments["nu20"] - Moments["nu02"] != 0:
                art = math.atan2(2 * Moments["nu11"], Moments["nu20"] - Moments["nu02"])
                theta = 0.5 * art
                theta = abs(theta)
                return theta
            return None

        for Scena in FM.Images:
            dep_front = Scena.Depth_front.copy()
            # Masking Part
            mask = self.Masquerade.Mask(Scena.Mask_front, dep_front)
            Out = self.Interaction.Class_One_Image(Scena.RGB_front, Scena.Depth_front, mask, None)
            Scena.Skeleton = self.Interaction.prvs_skel
            Scena.Values["Canvas"] = self.Interaction.prvs_canvas
            if Out is not None:
                Hand_patch_r, Hand_patch_d, Class, Center, p = Out
                Hand_angle = angle(mask,Center)
                Scena.Values["Mask"] = mask
                Scena.Values["Interaction_recognized"] = Class
                Scena.Values["Hand_Pos"] = Center
                Scena.Values["Interaction_posibility"] = p
                Scena.Values["Hand_Angle"] = Hand_angle
                self.Evaluator.add_data(Class,p)
        Result_n, Result_p, Class_p, Class_n, Total =self.Evaluator.calculate_output()
        FM.Values["Class_N"]=Class_n
        FM.Values["Result_N"]= Result_n
        FM.Values["Total_N"]= Total
        print "En caso de usar el voto normal ha salido "+ Class_n + " con "+Result_n.__str__()+" de votos de un total de "+Total.__str__()+" votos."
        print "En caso de usar el voto porcentaje ha salido " + Class_p + " con un porcentaje medio de " + Result_p.__str__() + "."
