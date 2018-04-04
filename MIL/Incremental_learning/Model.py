import cv2
import numpy as np

class Model():

    def __init__(self,type,values,info,patch,U):
        self.type = type
        if type == 'hist':
            self.values = np.float32(values)
        else:
            self.values = values[0]
        self.label = None
        self.patch = patch
        self.U = U
        self.info = 0
        self.score = None

    def set_score(self,score):
        self.score = score
    def set_vote(self,vote):
        self.info += vote

    def get_score_class(self,clas,Flag = False):
        if self.type == 'hist':
            p = []
            min_d = 9999999999
            for example in clas:
                dist = cv2.compareHist(example.values, self.values, cv2.cv.CV_COMP_BHATTACHARYYA)
                if Flag:
                    p.append(dist)
                else:
                    if dist < min_d:
                        min_d = dist
            if Flag:
                return p
            else:
                return min_d
        elif self.type == 'points':
            best = []
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            if self.values is None:
                return len(best)
            for example in clas:
                if len(example.values) is None:
                    continue
                matches = flann.knnMatch(self.values, example.values,k=2)

                # store all the good matches as per Lowe's ratio test.
                good = []

                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good.append(m)

                if len(good) > 10 and len(good) > len(best):
                    best = good
            return len(best)

    def get_score_global(self,classes):
        if self.type == 'hist':
            scores = []
            i=0
            for clas in classes:
                score = self.get_score_class(clas,True)
                for d in score:
                    scores.append((d,i))
                i+=1
            scores = sorted(scores, key=lambda s: s[0])
            return scores
        elif self.type == 'points':
            scores = []
            i=0
            for clas in classes:
                score = self.get_score_class(clas)
                scores.append((score,i))
                i+=1
            scores = sorted(scores, key=lambda s: s[0],reverse=True)
            return scores

    def get_worse(self,list,Flag = False):
        # Given a list of index and scores obtain the one that should be erased
        if Flag:
            list_aux = sorted(list,key=lambda s:s[1],reverse=False)
            list_aux[0][0].set_vote(1)
            list_aux[1][0].set_vote(1)
            list_aux[-1][0].set_vote(-1)
            return list_aux
        if self.type == 'hist':
            list_aux = sorted(list,key=lambda s:s[1],reverse=False)
            return list_aux[0][0]
        elif self.type == 'points':
            list_aux = sorted(list,key=lambda s:s[1],reverse=False)
            return list_aux[0][0]
