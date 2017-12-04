import cv2
import numpy as np

class Descriptors:

    def ComputeSIFT(self,RGB,Depth):
        """
        :param RGB:
        :param Depth:
        :return:
        """
        # mask = np.ma.masked_less_equal(Depth,30)
        # ones = np.zeros(RGB.shape,RGB.dtype)
        # final = np.where(mask==0,ones,RGB)
        # final = self.SuperPixel(final)
        final = RGB
        img1 = cv2.cvtColor(final,cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT()
        kp, descritors = sift.detectAndCompute(img1,None)
        self.n += 1
        if len(kp) == 0:
            self.num += 1
            return np.zeros((1, 128), np.float32), final
            # apply the Hellinger kernel by first L1-normalizing and taking the
            # square-root
        descritors /= (descritors.sum(axis=1, keepdims=True) + self.eps)
        descritors = np.sqrt(descritors)
        return descritors, kp

    def ComputeSURF(self,RGB,Depth):
        """
        :param RGB:
        :param Depth:
        :return:
        """
        # mask = np.ma.masked_less_equal(Depth,30)
        # ones = np.zeros(RGB.shape,RGB.dtype)
        # final = np.where(mask==0,ones,RGB)
        # final = self.SuperPixel(RGB)
        final = RGB
        img1 = cv2.cvtColor(final,cv2.COLOR_BGR2GRAY)
        surf = cv2.SURF(100)
        self.n += 1
        kp, descritors = surf.detectAndCompute(img1,None)
        if len(kp) == 0:
            self.num += 1
            return np.zeros((1, 128), np.float32), final
        return descritors,kp

    def ComputeBRISK(self,RGB,Depth):
        Gray =cv2.cvtColor(RGB,cv2.COLOR_BGR2GRAY)
        detector = cv2.BRISK(thresh=2, octaves=3,)
        scene_keypoints = detector.detect(Gray)
        scene_keypoints, scene_descriptors = detector.compute(Gray, scene_keypoints)
        img2 = cv2.drawKeypoints(RGB,scene_keypoints,None,(255,0,0),4)
        if len(scene_keypoints) == 0:
            self.num += 1
            return None, img2
        return scene_descriptors,scene_keypoints

    def ComputeORB(self, RGB, Depth):
        Gray = cv2.cvtColor(RGB, cv2.COLOR_BGR2GRAY)
        detector = cv2.ORB(nfeatures=600,scaleFactor=1.2,nlevels=4,edgeThreshold=5,firstLevel=0,WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE,patchSize=5)
        scene_keypoints = detector.detect(Gray)
        scene_keypoints, scene_descriptors = detector.compute(Gray, scene_keypoints)
        if len(scene_keypoints) == 0:
            self.num += 1
            return None, None
        return scene_descriptors, scene_keypoints

    def ComputeHC(self,RGB,Depth):
        B,G,R = cv2.split(RGB)
        B = B/32
        G = G/32
        R = R/32
        HistB = np.unique(B,return_counts = True)[1]
        HistG = np.unique(G, return_counts=True)[1]
        HistR = np.unique(R,return_counts = True)[1]
        output = np.concatenate((HistB, HistG, HistR), axis=0)
        output = (output * 100.0) / sum(output)
        return output, None

    def ComputeHC_deprecated(self,RGB,Depth):
        B,G,R = cv2.split(RGB)
        HistB = [0] * 8
        HistG = [0] * 8
        HistR = [0] * 8
        for i in xrange(8):
            HistB[i] = np.count_nonzero(B/32 == i)
            HistG[i] = np.count_nonzero(G/32 == i)
            HistR[i] = np.count_nonzero(R/32 == i)
        output = np.concatenate((HistB,HistG,HistR),axis=0)
        output = (output*100.0)/sum(output)
        return output,None

    def __init__(self):
        self.eps = 1e-7
        self.num = 0
        self.n = 0
