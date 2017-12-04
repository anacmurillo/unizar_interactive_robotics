from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.svm import SVC
import cv2
import os
import math
import matplotlib.pyplot as plt
import numpy as np


class NN():

    def dump(self,Path):
        joblib.dump((self.classes,self.labels,self.distance_min),Path,compress=3)
        return True

    def load(self,Path):
        self.path = Path
        if os.path.isfile(Path):
            self.classes, self.labels, self.distance_min = joblib.load(Path)
            return True
        return False


    ##############################################
    def reorganize(self,point1,weight):
        distance_global = 1000000
        index = 0
        value = None
        for i in xrange(len(point1)):
            p = np.float32(point1[i])
            point1_sub = point1[:]
            weight_sub = weight[:]

            if self.weight_flag:
                weight_sub.pop(i)
            point1_sub.pop(i)
            distance = self.distance(np.float32(point1_sub),weight_sub,p)
            if distance < distance_global :
                distance_global=distance
                index = i
                value = self.aux
        point1.pop(index)
        return point1,distance_global,value,index
        # output = [0]*1000
        # for i in xrange(1000):
        #     output[i]=(point1[i]+point2[i])/2
        # return output

    ############################################################
    def normalize_weights(self,class_i):
        sum = np.sum(self.weights[class_i])
        div = 1.0/sum
        for i in xrange(len(self.weights[class_i])):
            self.weights[class_i][i] *=div

    def weight_treatment(self,class_i,index,type,value=None):
        def comp(x1,x2):
            if len(x1)!= len(x2):
                return False
            else:
                for j in xrange(len(x1)):
                    if np.float32(x1[j])!=np.float32(x2[j]):
                        return False
                return True
        # print "Clase : "+class_i.__str__()+" Action: "+type+" Longitud: "+len(self.weights[class_i]).__str__()
        if type == "add":
            base = 1.0/(len(self.weights[class_i])+1)
            self.weights[class_i].append(base)
            self.normalize_weights(class_i)
        elif type == "reorp1":
            base = 1.0 /len(self.weights[class_i])
            self.weights[class_i].append(base)
        elif type == "reorp2":
            base = 1.0 / len(self.weights[class_i])
            i=-1
            for ind in xrange(len(self.classes[class_i])):
                if comp(self.classes[class_i][ind],value):
                    i = ind
                    break
            if i >=0:
                self.weights[class_i][i] = self.weights[class_i][i]+base
                self.weights[class_i].pop(index)
                self.normalize_weights(class_i)
        elif type == "test":
            base = 1.0 /len(self.weights[class_i])
            i=-1
            for ind in xrange(len(self.classes[class_i])):
                if comp(self.classes[class_i][ind],value):
                    i = ind
            if i >=0:
                ax = self.weights[class_i][i-1] + base
                self.weights[class_i][i-1] = ax
                self.normalize_weights(class_i)
        # print "Clase : "+class_i.__str__()+" Action: "+type+" Longitud: "+len(self.weights[class_i]).__str__()


    #######################################################
    def mean(self,hist):
        mean = 0.0
        # for i in hist:
        #     mean += i
        mean = sum(hist)
        mean /= len(hist)
        return mean

    def distance_classes(self,point):
        distances = []
        for i in xrange(len(self.classes)):
            d = self.distance(np.float32(self.classes[i]), self.weights[i], np.float32(point))
            distances.append(d)
        return distances

    def labeling(self,iclass):
        return self.labels[iclass]

    def classing(self, label):
        return self.labels.index(label)

    def distance(self,class_o,weights_o,point):
        dist = 0
        distance = 100000
        dist_min = 100000
        for i in xrange(len(class_o)):
            dis = self.distance_hist(class_o[i],point)
            if self.weight_flag:
                if not self.dist_flag:
                    dis = dis / weights_o[i]
                    if dis < distance:
                        distance = dis
                        self.aux = class_o[i]
                else:
                    dist += dis* weights_o[i]
                    if dis < dist_min:
                        dist_min = dis
                        self.aux = class_o[i]
                    distance = dist
            else:
                if dis < distance:
                    distance = dis
        return distance

    def distance_hist(self,point1,point2):
        return cv2.compareHist(point1,point2,cv2.cv.CV_COMP_BHATTACHARYYA)


    ###########################################################################
    def info(self,max):
        n_bins = 0
        # print "N de Clases:"
        # print self.classes.__len__()
        # print "Longitud de las Clases:"

        for i in xrange(len(self.classes)):
            n_bins += len(self.classes[i])
            if max:
                print "Clase :"+self.labels[i]+" Longitud: "+len(self.classes[i]).__str__()
                if self.weight_flag:
                    print "       Pesos: "+len(self.weights[i]).__str__()+ " Suma: "+sum(self.weights[i]).__str__()
                    print self.weights[i]
        print " Total Datos guardados: "+n_bins.__str__()
        return " Total Datos guardados: "+n_bins.__str__()
        # cols = int(np.ceil(self.classes.__len__()/2.0))
        # if cols == 1:
        #     fig, axes = plt.subplots(nrows=2, ncols=2)
        # else:
        #     fig, axes = plt.subplots(nrows=cols, ncols=cols)
        # ax = axes.flat
        # for i in xrange(len(self.classes)):
        #     ax[i].hist(range(0,self.size),bins=101, normed=0, histtype='bar',weights=self.classes[i])
        #     ax[i].set_title(self.labels[i])
        # plt.tight_layout()
        # plt.show()


    #################################################################################
    def train(self,  des, names):
        if len(self.classes) <= 0:
            self.classes.append([])
            self.classes[len(self.classes) - 1].append(des)
            self.weights.append([])
            self.weights[len(self.weights) - 1].append(1.0)
            self.labels.append(names)
            return -1
        else:
            label = 0
            if names in self.labels:
                i= self.labels.index(names)
                if len(self.classes[i]) < self.max_hist:
                    self.classes[i].append(des)
                    if self.weight_flag:
                        self.weight_treatment(i, len(self.classes[i]), "add")
                else:
                    self.classes[i].append(des)
                    if self.weight_flag:
                        self.weight_treatment(i, None, "reorp1")
                    out,d,value,index = self.reorganize(self.classes[i],self.weights[i])
                    if self.weight_flag:
                        self.weight_treatment(i,index,"reorp2",value=value)
                    self.classes[i] = out
                return 1
            else:
                self.classes.append([])
                self.classes[len(self.classes)-1].append(des)
                self.weights.append([])
                self.weights[len(self.weights) - 1].append(1.0)
                self.labels.append(names)
                return 0


    ##############################################################################
    def test(self,des, Number):
        if len(self.classes) <= 0:
            return -1,"",0
        else:
            distance_max = 1000000000
            label = 0
            aux = None
            for i in xrange(len(self.classes)):
                d = self.distance(np.float32(self.classes[i]),self.weights[i],np.float32(des))
                if Number == 1:
                    print d.__str__()+" "+self.labels[i]
                if distance_max > d:
                    distance_max = d
                    label = i
                    aux = self.aux
            if self.weight_flag:
                self.weight_treatment(label,None,"test",value=aux)
            if distance_max < self.distance_min:
                return 1,self.labels[label],distance_max
            else:
                return 0, self.labels[label],distance_max

    def __init__(self,distance,size,max_hist):
        self.aux = 0
        self.classes =[]
        self.weights = []
        self.labels = []
        self.size = size
        self.dist_flag = False
        self.weight_flag = False
        self.max_hist = max_hist
        self.distance_min = distance