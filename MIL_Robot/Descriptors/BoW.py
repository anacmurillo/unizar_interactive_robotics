from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.svm import SVC
import cv2
import os
import numpy as np

class BoW():


    def loadBoW(self,path):
        self.path=path
        if os.path.isfile(path):
            self.voc,self.k = joblib.load(path)
            return True
        return False

    def trainVoq(self,descriptors,n):
        self.voc, variance = kmeans(descriptors, n, 1)
        joblib.dump((self.voc,n),self.path,compress=3)
        return variance


    def testwoBoW(self, des_list, n):
        test_features = np.zeros((n, self.k), "float32")
        for i in xrange(n):
            words, distance = vq(des_list[i][1],self.voc)
            for w in words:
                test_features[i][w] += 1
        x = sum(sum(test_features))
        test_features = (test_features/x)*100
        return test_features

    def __init__(self):
        self.k = 1000
        self.voc = None
        self.loaded = False
