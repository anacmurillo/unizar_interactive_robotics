from sklearn.naive_bayes import GaussianNB
import numpy as np

class GNB():

    def __init__(self,):
        self.clf =  GaussianNB()
        self.labels = []
    def dump(self,Path):
        return None
    def load(self,Path):
        return None
    def label(self,index):
        return self.labels[index]

    def train(self,descriptor,label):
        print np.array(descriptor).reshape(1,-1)
        print np.array([label])
        if label not in self.labels:
            self.labels.append(label)
        self.clf.partial_fit(np.array(descriptor).reshape(1,-1),np.array([label]),self.labels)
        return True

    def test(self,descriptor):
        out = self.clf.predict(descriptor)
        return self.label(out)

    def ntest(self,descriptor,n):
        out = self.clf.predict_proba(descriptor)
        output= zip(self.labels,out)
        output.sort(key = lambda x: x[1],reverse = True)
        return [o[0] for o in output[:n]]


