import numpy as np
import cv2
import timeit
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb,slic
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb
import multiprocessing
import random
import math
import Obj_segment.Rect
import Candidate

class Object_Cand():

    def __init__(self):
        self.homo = np.array(
            [[9.94775973e-01, -4.09621743e-01, -4.37893262e+01], [-2.06444142e-02, 2.43247181e-02, 1.94859521e+02],
             [-1.13045909e-04, -1.41217334e-03, 1.00000000e+00]])
        self.centro = (0, 0)
        self.angulo = 0

    def inside(self,img,seg,p1,p2):
        D1 = img.copy()
        D2 = img.copy()
        D1 = D1[p1[1]:p2[1],p1[0]:p2[0]]
        D2[D2 ==seg ] = 0
        D2[D2 != 0] = -1
        D2 +=1
        D1[D1 ==seg ] = 0
        D1[D1 != 0] = -1
        D1 +=1
        Sum1= sum(sum(D1))
        Sum2= sum(sum(D2))
        Sum3= (p2[1]-p1[1])*(p2[0]-p1[0])
        if Sum1>int(0.30*Sum2) or Sum3*0.75<=Sum1:
            return True
        return False


    def bbox2(self,img_sp,sp,p1,p2):
        im = img_sp.copy()
        for seg in sp:
            if seg ==1:
                continue
            if self.inside(img_sp,seg,p1,p2):
                im[im == seg] = 0
        im[im!= 0] = -1
        im+=1
        if sum(sum(im))<20:
            return p1[1],p2[1],p1[0],p2[0]
        s = 0
        rmin = 0
        rmax = 639
        while (s == 0 and rmin < 639):
            s = sum(im[:, rmin])
            rmin += 1
        s = 0
        while (s == 0 and rmax > rmin):
            s = sum(im[:, rmax])
            rmax -= 1
        s = 0
        cmin = 0
        cmax = 479
        while (s == 0 and cmin <479):
            s = sum(im[cmin, :])
            cmin += 1
        s = 0
        while (s == 0 and cmax > cmin):
            s = sum(im[cmax, :])
            cmax -= 1
        return cmin , cmax , rmin, rmax

    def rotateImage(self,image, angle,center):
        rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
        result = cv2.warpAffine(image,rot_mat,image.shape[1::-1],flags=cv2.INTER_LINEAR)
        return result

    def Homo_get(self,x,y,inverted=False):
        p1 = [float(x), float(y), 1.0]
        p1 = np.array(p1)
        if inverted:
            r = np.dot(p1,np.linalg.inv(self.homo))
        else:
            r = np.dot(self.homo, p1)
        r = r / r[2]
        return r

    def get_images(self,img1, Mask1, img2, Mask2):

        #Process the Mask of the top camera
        Mask_1 = Mask1.copy()
        kernel = np.ones((7, 7), np.uint8)
        Mask_1 = cv2.dilate(Mask_1, kernel, 1)
        kernel = np.ones((4, 4), np.uint8)
        Mask_1 = cv2.erode(Mask_1, kernel, 1)
        edged = cv2.Canny(Mask_1,1,240)

        #Obtain the total of pixels in the mask
        total = sum(sum(Mask_1[:, :, 0]))

        #Find the biggest countour (The table)
        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_cc = max(cnts, key = cv2.contourArea)
        if cv2.contourArea(max_cc) <0.6*total:
            return None,None,None

        #Create a bounding box for the Table and rotate the images so the angle is 0
        x, y, w, h = cv2.boundingRect(max_cc)
        rect = cv2.minAreaRect(max_cc)
        angle =rect[2]
        Mask_1 = self.rotateImage(Mask_1, angle, (x + w / 2, y + h / 2))
        p_c = (x + w / 2, y + h / 2)

        #Reduce the area of search to the rotated table
        x_ini = x
        y_ini = y
        Mask_1 = Mask_1[y:y+h,x:x+w]
        Mask_1[0:5, :, :] = 255
        Mask_1 = cv2.bitwise_not(Mask_1)
        edged = cv2.Canny(Mask_1, 50, 200)

        #Process the Frontal camera and obtain the SuperPixels
        i = 0
        img2 = img2[200:480,:,:]
        Mask2 = Mask2[200:480,:,0]
        kernel = np.ones((7, 7), np.uint8)
        Mask2 = cv2.dilate(Mask2, kernel, 1)
        kernel = np.ones((4, 4), np.uint8)
        Mask2 = cv2.erode(Mask2, kernel, 1)
        Mask2 = cv2.bitwise_not(Mask2)
        Sup1 = cv2.bitwise_and(img2,img2,mask=Mask2)
        Sup = cv2.cvtColor(Sup1,cv2.COLOR_BGR2RGB)
        segments_fz = slic(Sup, n_segments=100, compactness=10, sigma=2)
        segments_fz[Mask2<1] = -1
        segments_fz += 2
        w_t,h_t,d = Mask_1.shape
        #Obtain the contours for the objects in the top camera and proyect them into the frontal one
        #  and obtain the BB with the superpixels
        Contornos=[]
        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if cv2.contourArea(cnt) < 100 or w >= 0.5*w_t or h >= 0.5*h_t:
                continue
            r = self.Homo_get(x_ini +x,y_ini + y-20)
            p1 = (max(min(int(r[0]),639),0),max(min(int(r[1])-200,279),0))
            r = self.Homo_get(x_ini + x+w, y_ini + y +h-10)
            p2 = (max(min(int(r[0]),639),0),max(min(int(r[1])-200,279),0))
            sp = np.unique(np.array(segments_fz[p1[1]:p2[1],p1[0]:p2[0]]))
            if len(sp) == 0:
                None
            elif sp[0] ==[1] and len(sp)==1:
                print "Empty..."
            else:
                Contornos.append([(p1[0],p1[1]+200),(p2[0],p2[1]+200),(x_ini +x,y_ini+y),(x_ini +x+w,y_ini+y+h),i])
            i+=1
        return Contornos,angle,p_c


    def add_cnt(self,Cnts,cnt):
        #No Contours stored
        if len(Cnts)==0:
            Cnts.append([cnt,1])
        else:
            done= False
            #Look for the same contour
            for i in xrange(len(Cnts)):
                p1 = Cnts[i][0][0]
                p2 = cnt[0]
                p3 = cnt[1]
                p4 = Cnts[i][0][1]
                d1 = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                d2 = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                d3 = math.hypot(p3[0] - p4[0], p3[1] - p4[1])
                d4 = math.hypot(p3[0] - p4[0], p3[1] - p4[1])
                if abs(d1)<=20 and abs(d2)<=20 and abs(d3)<=20 and abs(d4)<=20:
                    Cnts[i][1]= Cnts[i][1] + 1
                    done=True
            #No Countour similar
            if not done:
                Cnts.append([cnt,1])


    def RecalculateBB(self,Total,F_RGB, F_mask):
        Output = []
        if F_RGB is None or F_mask is None:
            return Output
        M_top,M_front = F_mask
        R_top,R_front = F_RGB

        # Preprocess Mask front
        kernel = np.ones((7, 7), np.uint8)
        Mask2 = cv2.dilate(M_front[:, :, 0], kernel, 1)
        kernel = np.ones((4, 4), np.uint8)
        Mask2 = cv2.erode(Mask2, kernel, 1)
        Mask2 = cv2.bitwise_not(Mask2)

        # Preprocess Images Top and Front
        Sp_front = cv2.bitwise_and(R_front, R_front, mask=Mask2)
        Sp_front = cv2.cvtColor(Sp_front, cv2.COLOR_BGR2RGB)

        segments_fz = slic(Sp_front, n_segments=250, compactness=20, sigma=5)

        #Set Sp number correctly
        segments_fz[Mask2 < 1] = -1
        segments_fz += 2

        #Recalculate the BB of the Front using SuperPixels and create the Candidate Structure
        for i in xrange(len(Total)):
            P_front_1, P_front_2, P_top_1, P_top_2, n = Total[i][0]
            P1 = Obj_segment.Rect.Point(P_top_1[0],P_top_1[1])
            P2 = Obj_segment.Rect.Point(P_top_2[0],P_top_2[1])
            Rec_top = Obj_segment.Rect.Rect(P1, P2)
            x,y = P_front_1
            x2,y2 = P_front_2
            sp = np.array(segments_fz[y:y2, x:x2])
            sp = np.unique(sp)

            #No superpixels, maintains the same bounding box
            if len(sp) == 0:
                P1 = Obj_segment.Rect.Point(x, y)
                P2 = Obj_segment.Rect.Point(x2, y2)
                rec = Obj_segment.Rect.Rect(P1, P2)

            #One superpixels, mantains the same
            elif sp[0] == [1] and len(sp) == 1:
                P1 = Obj_segment.Rect.Point(x, y)
                P2 = Obj_segment.Rect.Point(x2, y2)
                rec = Obj_segment.Rect.Rect(P1, P2)

            #Calculate the new BB
            else:
                cmin, cmax, rmin, rmax = self.bbox2(segments_fz, sp, P_front_1,P_front_2)
                P1 = Obj_segment.Rect.Point(rmin, cmin)
                P2 = Obj_segment.Rect.Point(rmax, cmax)
                rec = Obj_segment.Rect.Rect(P1, P2)

            Pos_top = Rec_top.center()
            Pos_front = rec.center()
            Patch = R_front[rec.top:rec.bottom,rec.left:rec.right]
            C = Candidate.Candidate(Rec_top,Pos_top,rec,Pos_front,Patch)
            Output.append(C)
        return Output


    def get_candidate(self,FM):
        count = 0
        Total = []
        F_RGB = None
        F_mask = None

        for f in FM.Images:
            RGB1 = f.RGB_top.copy()
            Depth1 = f.Depth_top.copy()
            Mask1 = f.Mask_top.copy()
            RGB2 = f.RGB_front.copy()
            Mask2 = f.Mask_front.copy()

            R, angle, p_c = self.get_images(RGB1, Mask1, RGB2, Mask2)
            if R is None:
                continue

            RGB2 =f.Rotated_RGB = self.rotateImage(RGB2, angle, p_c)
            f.Rotated_Depth = self.rotateImage(Depth1, angle, p_c)
            Mask2  = f.Rotated_Mask = self.rotateImage(Mask2, angle, p_c)

            if count == 0:
                F_RGB = (RGB1,RGB2)
                F_mask = (Mask1,Mask2)

            for K in xrange(len(R)):
                self.add_cnt(Total, R[K])

            if count > 5:
                break
            count += 1

        Total = [Values for Values in Total if Values[1] >= 4]
        Total = self.RecalculateBB(Total,F_RGB,F_mask)
        return Total
