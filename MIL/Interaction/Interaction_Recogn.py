import math
import os
import timeit
import Caffe2
import cv2
from NMS import nms
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
        self.save_path_r ="/home/iglu/catkin_ws/src/IL-pipeline/src/Hands/RGB/"
        self.save_path_d ="/home/iglu/catkin_ws/src/IL-pipeline/src/Hands/Depth/"
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
        self.clf_HC = SVM.SVM(self.model_path_HC, 200, 5000000, False, True)
        self.clf_HOG = SVM.SVM(self.model_path_HOG, 200, 5000000, False, True)
        self.n = 0
        self.trained = False
        # self.training2()
        if self.clf_HC.loaded:
            self.trained = True

    def clean(self):
        self.prvs_cnt = None
        self.prvs_per = None
        self.prvs_angle = None


    def training2(self):
        self.samples_HOG =[]
        self.labels_HOG=[]
        self.samples_HC =[]
        self.labels_HC=[]
        self.n = 0
        self.names = []
        self.samples_p =[]
        self.samples_n = []
        self.samples_nu = []
        for i in os.listdir(self.save_path_r):
            if i.startswith("sample_p_"):
                sub = "_" + i[9:].split('_', 1)[-1]
                self.names.append(sub)
                self.samples_p.append((cv2.imread(self.save_path_r + i), np.load(self.save_path_d + i[:-3] + "npy")))
                self.n += 1
            elif i.startswith("sample_n_"):
                sub = "_" + i[9:].split('_', 1)[-1]
                self.names.append(sub)
                self.samples_n.append((cv2.imread(self.save_path_r + i), np.load(self.save_path_d + i[:-3] + "npy")))
                self.n += 1
        mask = np.ones((100,100),np.uint8)
        print "Calculating the descriptors for the positive samples and training the system"
        print len(self.samples_p)
        self.i=0
        for sample,dep in self.samples_p:
            fd,fd2 = self.ComputeDescriptors(sample,dep,mask,True)
            if self.i ==0:
                self.samples_HOG = fd
                self.samples_HC = fd2
            else:
                self.samples_HOG = np.vstack((self.samples_HOG, fd))
                self.samples_HC = np.vstack((self.samples_HC, fd2))
            self.i += 1
            self.labels_HC.append(0)
            self.labels_HOG.append(0)

        print "Calculating the descriptors for the negative samples and training the system"
        print len(self.samples_n)
        for sample,dep in self.samples_n:
            fd,fd2 = self.ComputeDescriptors(sample,dep,mask,True)
            if self.i ==0:
                self.samples_HOG = fd
                self.samples_HC = fd2
            else:
                self.samples_HOG = np.vstack((self.samples_HOG, fd))
                self.samples_HC = np.vstack((self.samples_HC, fd2))
            self.i += 1
            self.labels_HC.append(1)
            self.labels_HOG.append(1)

        self.clf_HC.train(self.samples_HC,None,self.labels_HC,0,True)
        self.clf_HOG.train(self.samples_HOG,None,self.labels_HOG,0,True)
        self.trained = True


    def training(self):
        self.samples_HOG =[]
        self.labels_HOG=[]
        self.samples_HC =[]
        self.labels_HC=[]
        self.n = 0
        self.names = []
        self.samples_p =[]
        self.samples_n = []
        self.samples_nu = []
        for i in os.listdir(self.save_path_r):
            if i.startswith("sample_p_"):
                sub = "_" + i[9:].split('_', 1)[-1]
                self.names.append(sub)
                self.samples_p.append((cv2.imread(self.save_path_r + i), np.load(self.save_path_d + i[:-3] + "npy")))
                self.n += 1
            elif i.startswith("sample_n_"):
                sub = "_" + i[9:].split('_', 1)[-1]
                self.names.append(sub)
                self.samples_n.append((cv2.imread(self.save_path_r + i), np.load(self.save_path_d + i[:-3] + "npy")))
                self.n += 1
            elif i.startswith("sample_nu_"):
                sub = "_" + i[9:].split('_', 1)[-1]
                self.names.append(sub)
                self.samples_nu.append((cv2.imread(self.save_path_r + i), np.load(self.save_path_d + i[:-3] + "npy")))
                self.n += 1
        mask = np.ones((100,100),np.uint8)
        print "Calculating the descriptors for the positive samples and training the system"
        print len(self.samples_p)
        self.i=0
        for sample,dep in self.samples_p:
            fd,fd2 = self.ComputeDescriptors(sample,dep,mask,True)
            if self.i ==0:
                self.samples_HOG = fd
                self.samples_HC = fd2
            else:
                self.samples_HOG = np.vstack((self.samples_HOG, fd))
                self.samples_HC = np.vstack((self.samples_HC, fd2))
            self.i += 1
            self.labels_HC.append(1)
            self.labels_HOG.append(0)

        print "Calculating the descriptors for the negative samples and training the system"
        print len(self.samples_n)
        for sample,dep in self.samples_n:
            fd,fd2= self.ComputeDescriptors(sample,dep,mask,True)
            if self.i ==0:
                self.samples_HOG = fd
                self.samples_HC = fd2
            else:
                self.samples_HOG = np.vstack((self.samples_HOG, fd))
                self.samples_HC = np.vstack((self.samples_HC, fd2))
            self.i += 1
            self.labels_HC.append(1)
            self.labels_HOG.append(1)

        print "Calculating the descriptors for the Null samples and training the system"
        print len(self.samples_nu)
        for sample,dep in self.samples_nu:
            fd,fd2 = self.ComputeDescriptors(sample,dep,mask,True)
            if self.i ==0:
                # self.samples_HOG = fd
                self.samples_HC = fd2
            else:
                # self.samples_HOG = np.vstack((self.samples_HOG, fd))
                self.samples_HC = np.vstack((self.samples_HC, fd2))
            self.i += 1
            self.labels_HC.append(0)
            # self.labels_HOG.append(0)
        # np.save("Samples_Hand.npy",self.samples)
        # np.save("Labels_Hand.npy",self.labels)
        print len(self.samples_HC)
        self.clf_HC.train(self.samples_HC,None,self.labels_HC,0,True)
        self.clf_HOG.train(self.samples_HOG,None,self.labels_HOG,0,True)
        self.trained = True







    def Class_One_Image(self, image, depth, mask,mov,left=False, skeleton=None):

        def calc_point(arm, hand):
            theta = math.atan2(hand[1] - arm[1],
                               hand[0] - arm[0])
            if theta <0:
               theta = 2*math.pi+theta
            x2 = hand[1] + 1 * np.sin(theta)
            y2 = hand[0] + 1 * np.cos(theta)
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
            print "No Hand found"
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
        # Hand_patch_r = image[x - 50:x + 50, y - 50:y + 50, :].copy()
        # Hand_patch_d = depth[x - 50:x + 50, y - 50:y + 50].copy()
        # Hand_patch_m = mask[x - 50:x + 50, y - 50:y + 50].copy()
        # Hand_patch_move = mov[x - 50:x + 50, y - 50:y + 50].copy()
        Hand_patch_r = image[x - 100:x + 100, y - 100:y + 100, :].copy()
        Hand_patch_d = depth[x - 100:x + 100, y - 100:y + 100].copy()
        Hand_patch_m = mask[x - 100:x + 100, y - 100:y + 100].copy()
        Hand_patch_move = mov[x - 100:x + 100, y - 100:y + 100].copy()
        Hand_patch_r =cv2.resize(Hand_patch_r,None,fx=0.5,fy=0.5)
        Hand_patch_d =cv2.resize(Hand_patch_d,None,fx=0.5,fy=0.5)
        Hand_patch_m =cv2.resize(Hand_patch_m,None,fx=0.5,fy=0.5)
        Hand_patch_move =cv2.resize(Hand_patch_move,None,fx=0.5,fy=0.5)
        if np.count_nonzero(np.array(Hand_patch_move[:,:])) > 400:
            if self.window is None:
                return None
            self.window.append(1)
            if len(self.window)<5:
                return None
            fd, fd2 = self.ComputeDescriptors(Hand_patch_r, Hand_patch_d, Hand_patch_m, True)
            pred_hog = self.clf_HOG.predict(fd.reshape(1, -1))
            pred_hc = self.clf_HC.predict(fd2.reshape(1, -1))
            pred = (pred_hog[0] + pred_hc[0]) / 2.0
            pred = pred.tolist()
            p = max(pred)
            pred = pred.index(p)
            if pred == 0 and p > self.th:  # Point
                return ('Point', (x, y), p)
            elif pred == 1 and p > self.th:  # Show
                return ('Show', (x, y), p)
            else:
                return None
        else:
            # print "No Mov"
            if self.window is None:
                return None
            if len(self.window) >= 5:
                self.window = None
            return None

    def Class_One_Image_sliding(self,image,depth,mask,skeleton=None):

        def sliding_window(image,depth,mask, window_size, step_size):
            '''
            This function returns a patch of the input image `image` of size equal
            to `window_size`. The first image returned top-left co-ordinates (0, 0)
            and are increment in both x and y directions by the `step_size` supplied.
            So, the input parameters are -
            * `image` - Input Image
            * `window_size` - Size of Sliding Window
            * `step_size` - Incremented Size of Window
            The function returns a tuple -
            (x, y, im_window)
            where
            * x is the top-left x co-ordinate
            * y is the top-left y co-ordinate
            * im_window is the sliding window image
            '''
            for x in xrange(0, image.shape[0]-window_size[0], step_size[1]):
                for y in xrange(0, image.shape[1]-window_size[1], step_size[0]):
                    yield (x, y, image[y:y + window_size[1], x:x + window_size[0]],depth[y:y + window_size[1], x:x + window_size[0]],mask[y:y + window_size[1], x:x + window_size[0]])

        def calc_point(arm,hand):
            theta = math.atan2(hand[0] - arm[0],
                               hand[1] - arm[1])
            #if theta <0:
            #    theta = 2*math.pi+theta
            x2 = hand[1] + 1 * np.sin(theta)
            y2 = hand[0] + 1 * np.cos(theta)
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
            canvas, skeleton = self.skeleton.get_skeleton(image,self.scale)
            self.prvs_canvas = canvas
            self.prvs_skel = skeleton
        else:
            self.prvs_skel = skeleton
            self.prvs_canvas = None
        if Check_arm_right(skeleton):
            x,y = calc_point(skeleton['arm right'],skeleton['hand right'])
        elif Check_arm_left(skeleton):
            x, y = calc_point(skeleton['arm left'], skeleton['hand left'])
        else:
            print "No Hand found"
            return None
        im = image[x-100:x+100,y-100:y+100,:].copy()
        d = depth[x - 100:x + 100, y - 100:y + 100].copy()
        m = mask[x - 100:x + 100, y - 100:y + 100].copy()
        min_wdw_sz = (100, 100)
        step_size = (10, 10)
        detections_p = []
        detections_s = []
        for (x_n, y_n, Hand_patch_r,Hand_patch_d,Hand_patch_m) in sliding_window(im,d,m, min_wdw_sz, step_size):
            fd, fd2 = self.ComputeDescriptors(Hand_patch_r, Hand_patch_d , Hand_patch_m,True)
            pred = self.clf_HOG.predict(fd.reshape(1, -1))
            #pred_hc = self.clf_HC.predict(fd2.reshape(1, -1))
            #pred = (pred_hog[0]+pred_hc[0])/2.0
            pred = pred[0].tolist()
            p = max(pred)
            pred = pred.index(p)
            if pred == 0:
                detections_p.append((pred,p,(x+x_n,y+y_n)))
            else:
                detections_s.append((pred,p,(x+x_n,y+y_n)))
        if len(detections_p) >= len(detections_s):
            detections = sorted(detections_p, key=lambda detections: detections[1],
                reverse=True)
        else:
            detections = sorted(detections_s, key=lambda detections: detections[1],
                reverse=True)
        pred,p,(x,y) = detections[0]
        if pred == 0 and p > self.th: #Point
            return ('Point',(x,y),p)
        elif pred == 1 and p > self.th: #Show
            return ('Show',(x,y),p)
        else:
            return None

###############################################################################################
    def sliding_window(self, image, window_size, step_size):
        '''
        This function returns a patch of the input image `image` of size equal
        to `window_size`. The first image returned top-left co-ordinates (0, 0)
        and are increment in both x and y directions by the `step_size` supplied.
        So, the input parameters are -
        * `image` - Input Image
        * `window_size` - Size of Sliding Window
        * `step_size` - Incremented Size of Window
        The function returns a tuple -
        (x, y, im_window)
        where
        * x is the top-left x co-ordinate
        * y is the top-left y co-ordinate
        * im_window is the sliding window image
        '''
        for y in xrange(0, image.shape[0], step_size[1]):
            for x in xrange(0, image.shape[1], step_size[0]):
                yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


    def search_one_image(self, image, dep, mask):
        def inside(T, pt):

            def sign(p1, p2, p3):
                return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

            b1 = sign(pt, T[0], T[1]) < 0.0
            b2 = sign(pt, T[1], T[2]) < 0.0
            b3 = sign(pt, T[2], T[0]) < 0.0
            return ((b1 == b2) and (b2 == b3))

        def cirinside(p1, p2, r):
            if p1 == None:
                return False
            a1 = p2[0] - p1[0]
            a2 = p2[1] - p1[1]
            r2 = math.pow(r, 2)
            a1 = math.pow(a1, 2)
            a2 = math.pow(a2, 2)
            if a1 + a2 <= r2:
                return False
            else:
                return True

        start_time = timeit.default_timer()
        im = image.copy()
        min_wdw_sz = (100, 100)
        step_size = (10, 10)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faceCascade = cv2.CascadeClassifier("face_cascada.xml")
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                             flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
        y1 = 220
        y2 = 420
        for (x, y, w, h) in faces:
            aux = x
            x = y
            y = aux
            m1 = (240 - x) / (0 - y)
            m2 = (240 - x + h) / (640 - y + w)
            y1 = -m1 * x + y
            y2 = -m2 * (x + h) + (y + w)
            y1 = int(y1)
            y2 = int(y2)
        p1 = (0, 0)
        p2 = (y1, 0)
        p3 = (0, 240)
        T1 = (p1, p2, p3)
        p1 = (y2, 0)
        p2 = (640, 0)
        p3 = (640, 240)
        T2 = (p1, p2, p3)
        p_max = 0
        # Display the resulting frame
        # Load the classifier
        # List to store the detections
        n_windows = 0
        detections = []
        resized = im.copy()
        cloned = resized.copy()
        globalmask = image.copy()
        globalmask[globalmask == globalmask] = 0
        # The current scale of the image
        # Downscale the image and iterate
        C = None
        c = None
        for (x, y, im_window) in self.sliding_window(im, min_wdw_sz, step_size):
            m = np.array(mask[y:y + 100, x:x + 100])
            st = np.sum(np.sum(m)) / (100.0)
            # print st
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0] or inside(T1, (x, y)) or inside(
                    T2, (x + 100, y)) or st < 10.0 or cirinside(self.prvs_cnt, (x, y), 200):
                continue
            globalmask[y:y + 100, x:x + 100] = (255, 255, 255)
            fd, fd2 = self.ComputeDescriptors(im_window, dep[y:y + 100, x:x + 100], mask[y:y + 100, x:x + 100], False)
            pred = self.clf_HC.predict(fd2.reshape(1, -1))
            n_windows += 1
            if pred[0][1] >= self.th:
                Moments = cv2.moments(mask[y:y + 100, x:x + 100])
                # HuMoments = cv2.HuMoments(Moments)
                p1 = pred[0][1]
                # print "Encontrada mano en "+x.__str__()+"-"+y.__str__()
                # detections_p.append((x, y, p,int(min_wdw_sz[0] ),int(min_wdw_sz[1])))
                fd, fd2 = self.ComputeDescriptors(im_window, dep[y:y + 100, x:x + 100], mask[y:y + 100, x:x + 100],
                                                     True)
                pred = self.clf_HOG.predict(fd.reshape(1, -1))
                q = 1
                n = 0
                p = pred[0][0]
                while q <= 1:
                    if p < pred[0][q]:
                        n = q
                        p = pred[0][q]
                    q += 1
                pred = n
                if pred == 0 and p > self.th:
                    detections.append((x, y, p,
                                       int(min_wdw_sz[0]),
                                       int(min_wdw_sz[1]),
                                       pred, Moments, m))

                elif pred == 1 and p > self.th:
                    detections.append((x, y, p,
                                       int(min_wdw_sz[0]),
                                       int(min_wdw_sz[1]),
                                       pred, Moments, m))
        detections = nms(detections, 0.3)
        C = None
        # Display the results after performing NMS
        for (x_tl, y_tl, c, w, h, clas, Moments, inzauma) in detections:
            if clas:
                C = "Show"
            else:
                C = "Point"
            if Moments["nu20"] - Moments["nu02"] != 0:
                art = math.atan2(2 * Moments["nu11"], Moments["nu20"] - Moments["nu02"])
                theta = 0.5 * art
                theta = abs(theta)
                if self.prvs_angle != None:
                    theta = (theta + self.prvs_angle) / 2.0
            cx = int(Moments['m10'] / Moments['m00'])
            cy = int(Moments['m01'] / Moments['m00'])
            self.prvs_cnt = (y_tl + cy, x_tl + cx)
            self.prvs_angle = theta
            prob = c
            break
        if C is None:
            return None
        return (C,self.prvs_cnt,prob)


