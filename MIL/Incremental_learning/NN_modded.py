# import sklearn.externals.joblib as joblib
import joblib
import Model
import os
import cv2
import random

class NN_new():

    def __init__(self,distance,max_hist,Mode= 'V'):
        self.aux = 0
        self.classes =[]
        self.mode = Mode
        self.Pos = 0
        self.labels = []
        self.max_hist = max_hist
        self.distance_min = distance

    def dump(self,Path):
        joblib.dump((self.classes, self.labels, self.distance_min), Path, compress=3)
        return True

    def load(self,Path):
        self.path = Path
        if os.path.isfile(Path):
            self.classes, self.labels, self.distance_min = joblib.load(Path)
            return True
        return False

    def label(self,index):
        return self.labels[index]

    def clas(self,label):
        return self.labels.index(label)

    def visualization(self,path):
        n_bins = 0
        for i in xrange(len(self.classes)):
            n_bins += len(self.classes[i])
        if path is not None:
            for i in xrange(len(self.classes)):
                n= 0
                print os.path.exists(path+"/"+self.labels[i])
                print path+"/"+self.labels[i]
                if not os.path.exists(path+"/"+self.labels[i]):
                    os.mkdir(path+"/"+self.labels[i])
                for j in xrange(len(self.classes[i])):
                    cv2.imwrite(path+"/"+self.labels[i]+"/"+self.labels[i]+"_"+n.__str__()+".jpg",self.classes[i][j].patch)
                    n+=1

        #    if max:
        #        print "Clase :"+self.labels[i]+" Longitud: "+len(self.classes[i]).__str__()
        #print " Total Datos guardados: "+n_bins.__str__()
        return n_bins

    def reorganize(self,clas):
        Scores = []
        i = 0
        if self.mode =='N':
            for c in clas:
                sub_clas = [data for data in clas[0:i]+clas[i+1:len(clas)]]
                score = c.get_score_class(sub_clas)
                Scores.append((i,score))
                i+=1
            index = clas[0].get_worse(Scores)
            new = [data for data in clas if data != clas[index]]
            return new, Scores[index], index
        elif self.mode=='V':
            for c in clas:
                sub_clas = [data for data in clas[0:i]+clas[i+1:len(clas)]]
                score = c.get_score_class(sub_clas)
                Scores.append((c,score))
                i+=1
            lista = clas[0].get_worse(Scores,True)
            if self.Pos == 5:
                self.Pos=0
                lista = sorted(lista, key= lambda s: s[0].info)
                new = [data for data in clas if data != lista[0]]
                return new, None, 0
            else:
                self.Pos+=1
                new = [data for data in clas if data != lista[0]]
                return new, None, 0
        else:
            index = random.randint(0,len(clas)-1)
            new = [data for data in clas[0:index]+clas[index+1:len(clas)]]
            return new,1,index


    def train(self,descriptor,label):
        if len(self.classes) <= 0:
            self.classes.append([])
            self.classes[len(self.classes) - 1].append(descriptor)
            self.labels.append(label)
            return -1
        else:
            if label in self.labels:
                i= self.labels.index(label)
                if len(self.classes[i]) < self.max_hist:
                    self.classes[i].append(descriptor)
                else:
                    self.classes[i].append(descriptor)
                    out,d,index = self.reorganize(self.classes[i])
                    self.classes[i] = out
                return 1
            else:
                self.classes.append([])
                self.classes[len(self.classes)-1].append(descriptor)
                self.labels.append(label)
                return 0

    def test(self,descriptor):
        if len(self.classes) <= 0:
            return -1,"",0
        else:
            distance_max = 1000000000
            label = 0
            for i in xrange(len(self.classes)):
                d = descriptor.get_score_class(self.classes[i])
                if distance_max < d:
                    distance_max = d
                    label = i
            if distance_max < self.distance_min:
                return 1,self.labels[label],distance_max
            else:
                return 0, self.labels[label],distance_max

    def ntest(self,descriptor,n):
        if len(self.classes) <= 0:
            return -1,"",0
        else:
            out = descriptor.get_score_global(self.classes)
            #print [(o[0],self.label(o[1])) for o in out]
            out = out[0:n]
            return [self.label(o[1]) for o in out],[o[0] for o in out]
