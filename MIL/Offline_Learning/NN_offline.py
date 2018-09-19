from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import DistanceMetric
import numpy as np
import os
import joblib
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import *


class NN_offline:

    def load(self,path):
        if os.path.isfile(path):
            self.clf, self.leaf_size, self.metric,self.names,self.clases = joblib.load(path)
            return True
        return False

    def train(self,descri,names):
        def distance_hist(point1, point2):
            return cv2.compareHist(np.array(point1,np.float32), np.array(point2,np.float32), cv2.cv.CV_COMP_BHATTACHARYYA)
        # unique, counts = np.unique(names, return_counts=True)
        # print dict(zip(unique, counts))
        # R = zip(descri,names)
        # sorted_by_second = sorted(R, key=lambda tup: tup[1])
        # descri = np.array(sorted_by_second)[:,0]
        # descri = np.array([D for D in descri])
        # # D = distance_hist(descri[0],descri[0])
        # Y = pdist(descri,'euclidean')
        # Y = squareform(Y)
        # Y = (Y/Y.max())*255
        # # np.save("Matrix_NN_DL.npy",Y)
        # Size_block = 2
        # Matri = np.zeros((Y.shape[0]*Size_block, Y.shape[1]*Size_block), np.float32)
        # for i in xrange(Y.shape[0]):
        #     for j in xrange(Y.shape[1]):
        #         Value = Y[i][j]
        #         Matri[i * Size_block:(i + 1) * Size_block, j * Size_block:(j + 1) * Size_block] = Value
        # plt.imsave("Matriz_Distancias.jpg",Matri,cmap='hot')
        # plt.show()
        self.clf = NearestNeighbors(3)
        self.clf.fit(descri)
        self.names = names
        self.clases = np.unique(self.names)
        joblib.dump((self.clf, self.leaf_size, self.metric,self.names,self.clases), self.path, compress=3)
        return self.names

    def get_names(self):
        return self.clases

    def IsObject(self,label):
        return label in self.clases

    def test(self,descri):
        names = []
        descri =  descri.reshape(-1, 1).transpose()
        for d in descri:
            d = d.reshape(-1, 1).transpose()
            dist, ind = self.clf.kneighbors(d)
            Out = []
            ind = ind[0]
            for n in xrange(len(ind)):
                i = ind[n]
                if Out == []:
                    Out.append((self.names[i],1))
                elif self.names[i] not in [t[0] for t in Out]:
                    Out.append((self.names[i],1))
                else:
                    k = [t[0] for t in Out].index(self.names[i])
                    Out[k]=(Out[k][0],Out[k][1]+1)
            m = max([t[1] for t in Out])
            ind = Out[[t[1] for t in Out].index(m)][0]
            names.append(ind)
        return names

    def __init__(self,path,leaf_size):
        self.path = path
        if not self.load(path):
            self.clf = None
            self.path = path
            self.clases = None
            self.leaf_size = leaf_size
            self.names = None
            self.metric = 'mahalanobis'