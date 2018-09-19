from Incremental_learning.NN import *
from Incremental_learning.Model import *
import numpy as np
import math

class Speak_seg():

    def __init__(self,Path):
        self.incremental = NN(0.1,20,20)
        print self.incremental.load(Path)
        self.candidates = []

    def add_candidates(self, Candidates):
        self.candidates = Candidates
        for c in self.candidates:
            # M = Model('hist', c.Descriptors["HC"], None, None)
            n,label,D = self.incremental.test(c.Descriptors["HC"],0)
            if c.patch_T is not None:
                # M = Model('hist', c.Descriptors_T["HC"], None, None)
                n, label2, D2 = self.incremental.test(c.Descriptors["HC"],0)
                if label2 ==label and D2 < D:
                    D = D2
                elif label2 != label and D2 < D:
                    label = label2
                    D = D2
            c.Label = label
            c.Values["D"] = D

    def obtain_candidate(self,speech):
        def dist(p1,p2):
            return math.hypot(p2[0] - p1[0],p2[1] - p1[1])
        Candidates = speech[1]
        direction = speech[3]
        reference = speech[5]
        D = 9999999
        c = None
        for candidate in self.candidates:
            if candidate.Label == reference and candidate.Values["D"] < D:
                c = candidate
                D = candidate.Values["D"]
        if c is None:
            print "Referencia no encontrada"
            return None,None
        center = c.BB_top.center()
        if direction == 'Up':
            distances = [(Cand,min(dist((Cand.BB_top.left,Cand.BB_top.bottom),center),dist((Cand.BB_top.right,Cand.BB_top.bottom),center))) for Cand in self.candidates if Cand.BB_top.bottom<=c.BB_top.top]
            if distances == []:
                distances = [(Cand, min(dist((Cand.BB_top.left, Cand.BB_top.bottom), center),
                                        dist((Cand.BB_top.right, Cand.BB_top.bottom), center))) for Cand in self.candidates]
            distances = sorted(distances,key=lambda distances: distances[1])
            Candidate = distances[0][0]
            Candidate.add_label(Candidates)
            return Candidate,c
        elif direction == 'Down':
            distances = [(Cand,min(dist((Cand.BB_top.left,Cand.BB_top.top),center),dist((Cand.BB_top.right,Cand.BB_top.top),center))) for Cand in self.candidates if Cand.BB_top.top>=c.BB_top.bottom]
            if distances == []:
                distances = [(Cand, min(dist((Cand.BB_top.left, Cand.BB_top.top), center),
                                        dist((Cand.BB_top.right, Cand.BB_top.top), center))) for Cand in self.candidates]
            distances = sorted(distances,key=lambda distances: distances[1])
            Candidate = distances[0][0]
            Candidate.add_label(Candidates)
            return Candidate,c
        elif direction == 'Right':
            distances = [(Cand,min(dist((Cand.BB_top.left,Cand.BB_top.bottom),center),dist((Cand.BB_top.left,Cand.BB_top.top),center))) for Cand in self.candidates if Cand.BB_top.left>=c.BB_top.right]
            if distances == []:
                distances = [(Cand, min(dist((Cand.BB_top.left, Cand.BB_top.bottom), center),
                                        dist((Cand.BB_top.left, Cand.BB_top.top), center))) for Cand in self.candidates]
                distances = sorted(distances,key=lambda distances: distances[1])
            Candidate = distances[0][0]
            Candidate.add_label(Candidates)
            return Candidate,c
        elif direction == 'Left':
            distances = [(Cand,min(dist((Cand.BB_top.right,Cand.BB_top.top),center),dist((Cand.BB_top.right,Cand.BB_top.bottom),center))) for Cand in self.candidates if Cand.BB_top.right<=c.BB_top.left]
            if distances == []:
                distances = [(Cand, min(dist((Cand.BB_top.right, Cand.BB_top.top), center),
                                        dist((Cand.BB_top.right, Cand.BB_top.bottom), center))) for Cand in self.candidates]
            distances = sorted(distances,key=lambda distances: distances[1])
            Candidate = distances[0][0]
            Candidate.add_label(Candidates)
            return Candidate,c
        else:
            print "Direction Error"
            return None,None
