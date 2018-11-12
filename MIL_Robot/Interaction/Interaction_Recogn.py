import math
import os
import timeit
import Caffe2
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
                          visualise=self.visualize)
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

    def __del__(self):
        del self.skeleton

    def __init__(self,sizeHC,th,Skeleton,str):
        if Skeleton is None:
              Skeleton = Caffe2.skeleton(str)
        self.model_path_HC = "Interaction/hand_HC_svm_"+sizeHC.__str__()+".model"
        self.model_path_HOG = "Interaction/hand_HOG_svm_"+sizeHC.__str__()+".model"
        self.sizeHC = sizeHC
        self.orientations = 12
        self.pixels_per_cell = [10, 10]
        self.cells_per_block = [10, 10]
        self.visualize = True
        self.normalize = True
        self.prvs_cnt = None
        self.th = th
        self.skeleton = Skeleton
        self.scale = [0.7]
        self.window = []
        self.prvs_skel = None
        self.prvs_angle = None
        self.prvs_canvas=None
        self.clf_HC = SVM.SVM(self.model_path_HC, 200, 50000, False, True)
        self.clf_HOG = SVM.SVM(self.model_path_HOG, 200, 50000, False, True)
        self.n = 0
        self.trained = False
        if self.clf_HC.loaded:
            self.trained = True

    def clean(self):
        self.prvs_cnt = None
        self.prvs_per = None
        self.prvs_angle = None



    def Class_One_Image(self, image, depth, mask,mov,left=False, skeleton=None):

        def calc_point(arm, hand):
            theta = math.atan2(hand[1] - arm[1],
                               hand[0] - arm[0])
            # print theta
            if theta <0:
               theta = 2*math.pi+theta
            x2 = hand[1] + 60 * np.sin(theta)
            y2 = hand[0] + 60 * np.cos(theta)
            if int(y2) > 380:
                y2 = 380
            elif int(y2) < 100:
                y2 = 100
            if int(x2) > 540:
                x2 = 540
            elif int(x2) < 100:
                x2 = 100
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
            # print "No Hand found"
            #return None
            canvas, skeleton = self.skeleton.get_skeleton(image, self.scale)
            self.prvs_canvas = canvas
            self.prvs_skel = skeleton
        else:
            self.prvs_skel = skeleton
            self.prvs_canvas = self.skeleton.draw_skeleton(image.copy(),skeleton)
        if Check_arm_right(skeleton) and not left:
            x, y = calc_point(skeleton['arm right'], skeleton['hand right'])
        elif Check_arm_left(skeleton) and left:
            x, y = calc_point(skeleton['arm left'], skeleton['hand left'])
        else:
            # print "No Hand found"
            return None
        Hand_patch_r = image[x - 100:x + 100, y - 100:y + 100, :].copy()
        Hand_patch_d = depth[x - 100:x + 100, y - 100:y + 100].copy()
        Hand_patch_m = mask[x - 100:x + 100, y - 100:y + 100].copy()
        Hand_patch_move = mov[x - 100:x + 100, y - 100:y + 100].copy()
        Hand_patch_r =cv2.resize(Hand_patch_r,None,fx=0.5,fy=0.5)
        Hand_patch_d =cv2.resize(Hand_patch_d,None,fx=0.5,fy=0.5)
        Hand_patch_m =cv2.resize(Hand_patch_m,None,fx=0.5,fy=0.5)
        Hand_patch_move =cv2.resize(Hand_patch_move,None,fx=0.5,fy=0.5)
        fd, fd2 = self.ComputeDescriptors(Hand_patch_r, Hand_patch_d, Hand_patch_m, True)
        pred_hog = self.clf_HOG.predict(fd.reshape(1, -1))
        pred = pred_hog[0].tolist()
        p = max(pred)
        pred = pred.index(p)
        if pred == 0 and p > self.th:  # Point
            return ('Point', (x, y), p)
        elif pred == 1 and p > self.th:  # Show
            return ('Show', (x, y), p)
        else:
            return None


