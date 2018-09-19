from scipy.cluster.vq import *
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.svm import *
from sklearn.metrics import *
from sklearn import linear_model
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


class SVM_offline:

    def load(self,path):
        if os.path.isfile(path):
            self.clf, self.clf.classes_, self.stdSlr = joblib.load(path)
            return True
        return False

    def predict(self,descriptors):
        if self.prob:
            P = self.clf.predict_proba(descriptors)
            # print P
            # print max(P[0])
            i =np.nonzero(P[0] == max(P[0]))[0][0]
            return self.clf.classes_[i],P[0][i]
        else:
            return self.clf.predict(descriptors)

    def get_names(self):
        return self.clf.classes_

    def IsObject(self,label):
        return label in self.clf.classes_

    def train(self, descriptors, names):
        # Scaling the words
        # N = Normalizer().fit(descriptors)
        # descriptors = N.transform(descriptors)
        # self.stdSlr = StandardScaler().fit(descriptors)
        # im_features = self.stdSlr.transform(descriptors)
        #
        # Train the Linear SVM
        # unique, counts = np.unique(names, return_counts=True)
        # print dict(zip(unique, counts))
        # C_range = np.logspace(-5, 10, 13)
        # gamma_range = np.logspace(-9, 5, 13)
        # param_grid = dict(gamma=gamma_range, C=C_range)
        # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
        # grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        # grid.fit(descriptors, names)
        #
        # print("The best parameters are %s with a score of %0.2f"
        #       % (grid.best_params_, grid.best_score_))
        self.clf = SVC(kernel='rbf', C=self.C, verbose=False, max_iter=self.iter,
                       probability=self.prob, gamma=self.gamma)
        names = names.ravel()
        self.clf.fit(descriptors, names)
        if self.prob:
            pred = self.clf.predict(descriptors)
            # print("Classification report for classifier %s:\n%s\n"
            #       % (self.clf, classification_report(names, pred)))
            # print("Confusion matrix:\n%s" % confusion_matrix(names, pred))
            # print self.clf.classes_
            # print self.clf.score(descriptors, names)
        # joblib.dump((self.clf, self.clf.classes_, self.stdSlr), self.path, compress=3)
        return self.clf.classes_


    def test(self, descriptors):
        # Scale the features
        # test_features = self.stdSlr.transform(descriptors)
        descriptors = descriptors.reshape(-1, 1).transpose()
        return self.predict(descriptors)


    def __init__(self, path, C,gamma, iter=2000, c_w=False, probability=True):
        if not self.load(path):
            self.path = path
            self.clf = SVC()
            self.C = C
            self.gamma = gamma
            self.iter = iter
            self.prob = probability
            if c_w:
                self.class_weight = {0: 1000, 1: 200}
            else:
                self.class_weight = 'balanced'
            self.stdSlr = None
            self.loaded = False
        else:
            self.path = path
            self.C = C
            self.gamma = gamma
            self.iter = iter
            self.prob = probability
            if c_w:
                self.class_weight = {0: 1000, 1: 200}
            else:
                self.class_weight = 'balanced'
            self.loaded = True