from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.svm import *
from sklearn.metrics import *
from sklearn import cross_validation
from sklearn import linear_model
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np


class SVM:

    def load(self,path):
        if os.path.isfile(path):
            self.clf, self.clf.classes_, self.stdSlr = joblib.load(path)
            return True
        return False

    def predict(self,descriptors):
        if self.prob:
            return self.clf.predict_proba(descriptors)
        else:
            return self.clf.predict(descriptors)

    def decision_function(self,descriptors):
        return self.clf.decision_function(descriptors)

    def train(self, descriptors,des_list,names, n,proba):
        def Cross_validation(clf,descriptors,names):
            n_samples = descriptors.shape[0]
            cv = cross_validation.ShuffleSplit(n_samples, n_iter=10,test_size = 0.3, random_state = 10)
            score = cross_validation.cross_val_score(clf, descriptors, names, cv=cv)
            print score
        def Cross_validation_per_class(clf,descriptors,names):
            cv = cross_validation.StratifiedShuffleSplit(names, n_iter=10,test_size = 0.3, random_state = 10)
            Score = []
            descriptors = np.array(descriptors)
            names = np.array(names)
            for train_index, test_index in cv:
                X_train, X_test = descriptors[train_index], descriptors[test_index]
                y_train, y_test = names[train_index], names[test_index]
                clf.fit(X_train,y_train)
                C0= [y_test==0]
                C1= [y_test==1]
                S0 = clf.score(X_test[C0], y_test[C0])
                S1 = clf.score(X_test[C1],y_test[C1])
                Score.append((S0,S1))
            for i in xrange(10):
                print Score[i][0].__str__()+";"+Score[i][1].__str__()
            # print Score
        # Scaling the words
        # self.stdSlr = StandardScaler().fit(descriptors)
        # im_features = self.stdSlr.transform(descriptors)

        # Train the Linear SVM
        self.clf = SVC(kernel='rbf',C=self.C,verbose=False,class_weight=self.class_weight,max_iter=self.iter,probability=self.prob,gamma=self.gamma)
        # Cross_validation_per_class(self.clf, descriptors, names)
        self.clf.fit(descriptors, names)
        if proba:
            pred = self.clf.predict(descriptors)
            # print("Classification report for classifier %s:\n%s\n"
            #       % (self.clf, classification_report(names, pred)))
            print("Confusion matrix:\n%s" % confusion_matrix(names, pred))
            # print self.clf.classes_
            print self.clf.score(descriptors,names)
        joblib.dump((self.clf, self.clf.classes_, self.stdSlr), self.path, compress=3)
        return self.clf.classes_

    def test(self, descriptors,des_list, n):
        # Scale the features
        test_features = self.stdSlr.transform(descriptors)

        return self.clf.predict(test_features)

    def __init__(self,path,C,iter,c_w,probability):
        if not self.load(path):
            self.path = path
            self.clf = SVC()
            self.C = C
            self.gamma =0.1
            self.iter = iter
            self.prob = probability
            if c_w :
                self.class_weight = {0:1000,1:200}
            else:
                self.class_weight = 'balanced'
            self.stdSlr = None
            self.loaded = False
        else:
            self.path = path
            self.C = C
            self.gamma = 0.1
            self.iter = iter
            self.prob = probability
            if c_w :
                self.class_weight = {0:1000,1:200}
            else:
                self.class_weight = 'balanced'
            self.loaded = True