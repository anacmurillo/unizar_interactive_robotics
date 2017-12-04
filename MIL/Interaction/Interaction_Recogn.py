import math
import os
import timeit
import CNN
import cv2
import numpy as np
from skimage.feature import hog

import SVM


class Interaction_Recogn:

    def ComputeDescriptors(self,RGB,Depth,dep_mask,h):
        dep = np.float32(Depth)
        mask=dep_mask.copy()
        if h:
            fd, imn = hog(dep, self.orientations, self.pixels_per_cell, self.cells_per_block,
                          self.visualize, self.normalize)
        else:
            fd = []
        fgrid = np.array([])
        for i in xrange(4):
            for j in xrange(4):
                sub = RGB[25*i:25*(i+1),25*j:25*(j+1)]
                sub_mask = mask[25*i:25*(i+1),25*j:25*(j+1)]
                fsub = self.ComputeHC(sub,sub_mask)
                fgrid = np.concatenate((fgrid,fsub))
        fd2 = fgrid.copy()
        return fd,fd2


    def ComputeHC(self,RGB,mask):
        B,G,R = cv2.split(RGB)
        HistB = [0] * self.sizeHC
        HistG = [0] * self.sizeHC
        HistR = [0] * self.sizeHC
        nt = 256 / self.sizeHC
        for i in xrange(self.sizeHC):
            HistB[i] = np.count_nonzero((B / nt == i) * mask)
            HistG[i] = np.count_nonzero((G / nt == i) * mask)
            HistR[i] = np.count_nonzero((R / nt == i) * mask)
        HistB = np.array(HistB)
        HistG = np.array(HistG)
        HistR = np.array(HistR)
        if (sum(HistB)) == 0:
            output = [0] * self.sizeHC * 3
            output = np.array(output)
        else:
            HistB = HistB * 1.0 / (sum(HistB))
            HistG = HistG * 1.0 / (sum(HistG))
            HistR = HistR * 1.0 / (sum(HistR))
            output = np.concatenate((HistB, HistG, HistR), axis=0)
        return output

    def __init__(self,sizeHC,th,Skeleton):
        if Skeleton is None:
            Skeleton = CNN.skeleton()
        self.model_path_HC = "Interaction/hand_HC_svm_"+sizeHC.__str__()+".model"
        self.model_path_HOG = "Interaction/hand_HOG_svm_"+sizeHC.__str__()+".model"
        self.sizeHC = sizeHC
        self.orientations = 12
        self.pixels_per_cell = [10, 10]
        self.cells_per_block = [10, 10]
        self.visualize = True
        self.normalize = True
        self.th = th
        self.skeleton = Skeleton
        self.scale = [0.7]
        self.prvs_skel = None
        self.clf_HC = SVM.SVM(self.model_path_HC, 200, 5000000, False, True)
        self.clf_HOG = SVM.SVM(self.model_path_HOG, 200, 5000000, False, True)
        self.n = 0
        self.trained = False
        if self.clf_HC.loaded:
            self.trained = True

    def clean(self):
        self.prvs_cnt = None
        self.prvs_per = None
        self.prvs_angle = None


    def Class_One_Image(self,image,depth,mask,skeleton=None):

        def calc_point(arm,hand):
            theta = math.atan2(hand[0] - arm[0],
                               hand[1] - arm[1])
            if theta <0:
                theta = 2*math.pi+theta
            x2 = hand[1] + 1 * np.sin(theta)
            y2 = hand[0] + 1 * np.cos(theta)
            return (int(y2), int(x2))

        def Check_arm_right(skeleton):
            if not ('hand right' in skeleton.keys() and 'arm right' in skeleton.keys()):
                return False
            else:
                return True

        def Check_arm_left(skeleton):
            if not ('hand left' in skeleton.keys() and 'arm left' in skeleton.keys()):
                return False
            else:
                return True

        if skeleton is None:
            canvas, skeleton = self.skeleton.get_skeleton(image,self.scale)
            self.prvs_canvas = canvas
            self.prvs_skel = skeleton
        else:
            self.prvs_skel = skeleton
        if Check_arm_right(skeleton):
            x,y = calc_point(skeleton['arm right'],skeleton['hand right'])
        elif Check_arm_left(skeleton):
            x, y = calc_point(skeleton['arm left'], skeleton['hand left'])
        else:
            print "No Hand found"
            return None
        Hand_patch_r = image[x-50:x+50,y-50:y+50,:].copy()
        Hand_patch_d = depth[x - 50:x + 50, y - 50:y + 50].copy()
        Hand_patch_m = mask[x - 50:x + 50, y - 50:y + 50].copy()
        fd, fd2 = self.ComputeDescriptors(Hand_patch_r, Hand_patch_d , Hand_patch_m,True)
        pred = self.clf_HOG.predict(fd.reshape(1, -1))
        pred = pred[0].tolist()
        p = max(pred)
        pred = pred.index(p)
        if pred == 0 : #Point
            return (Hand_patch_r,Hand_patch_d,'Point',(x,y),p)
        elif pred == 1 : #Show
            return (Hand_patch_r, Hand_patch_d, 'Show',(x,y),p)
        else:
            return None
