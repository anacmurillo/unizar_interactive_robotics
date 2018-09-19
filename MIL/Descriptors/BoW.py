from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.svm import SVC
import cv2
import os
import numpy as np

class BoW():

    def load(self,path):
        self.path = path
        if os.path.isfile(path):
            self.clf, self.clf.classes_, self.stdSlr, self.k, self.voc = joblib.load(path)
            return True
        return False

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


    def trainwBoW(self, descriptors, des_list, names, n):
        im_features = np.zeros((n, self.k), "float64")
        for i in xrange(n - 1):
            words, distance = vq(des_list[i][1], self.voc)
            for w in words:
                im_features[i][w] += 1

        # Perform Tf-Idf vectorization
        print im_features.shape
        np.savetxt("/home/iglu/Desktop/features.txt", im_features)
        nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
        idf = np.array(np.log((1.0 * n + 1) / (1.0 * nbr_occurences + 1)), 'float64')

        # Scaling the words
        self.stdSlr = StandardScaler().fit(im_features)
        im_features = self.stdSlr.transform(im_features)

        # Train the Linear SVM
        self.clf = SVC(probability=True)
        self.clf.fit(im_features, names)
        print self.clf.classes_
        joblib.dump((self.clf, self.clf.classes_, self.stdSlr, self.k, self.voc), self.path, compress=3)
        return self.clf.classes_

    def trainwoBoW(self, descriptors,des_list,names, n):
        self.voc, variance = kmeans(descriptors, self.k, 1)
        im_features = np.zeros((n, self.k), "float64")
        for i in xrange(n-1):
            words, distance = vq(des_list[i][1], self.voc)
            for w in words:
                im_features[i][w] += 1

        # Perform Tf-Idf vectorization
        print im_features.shape
        np.savetxt("/home/iglu/Desktop/features.txt",im_features)
        nbr_occurences = np.sum( (im_features > 0) * 1, axis=0)
        idf = np.array(np.log((1.0*n+1) / (1.0*nbr_occurences + 1)), 'float64')

        # Scaling the words
        self.stdSlr = StandardScaler().fit(im_features)
        im_features = self.stdSlr.transform(im_features)

        # Train the Linear SVM
        self.clf = SVC(probability = True)
        self.clf.fit(im_features, names)
        print self.clf.classes_
        joblib.dump((self.clf, self.clf.classes_, self.stdSlr, self.k, self.voc), self.path, compress=3)
        return self.clf.classes_


    def testwBoW(self, descriptors, des_list, n):
        test_features = np.zeros((n, self.k), "float64")
        for i in xrange(n):
            words, distance = vq(des_list[i][1], self.voc)
            for w in words:
                test_features[i][w] += 1

        # Perform Tf-Idf vectorization
        nbr_occurences = np.sum((test_features > 0) * 1, axis=0)
        idf = np.array(np.log((1.0 * n + 1) / (1.0 * nbr_occurences + 1)), 'float64')

        # Scale the features
        test_features = self.stdSlr.transform(test_features)

        return self.clf.predict_proba(test_features)

    def testwoBoW(self, des_list, n):
        test_features = np.zeros((n, self.k), "float32")
        for i in xrange(n):
            words, distance = vq(des_list[i][1],self.voc)
            for w in words:
                test_features[i][w] += 1
        x = sum(sum(test_features))
        test_features = (test_features/x)*100
        return test_features

    def __init__(self,path):
        if not self.load(path):
            self.k = 1000
            self.clf = SVC()
            self.stdSlr = None
            self.voc = None
            self.loaded = False
            self.path = path
        else:
            self.loaded = True
            self.path = path
