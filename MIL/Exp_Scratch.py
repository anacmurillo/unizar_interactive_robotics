import cv2
import cv
import sys
import os
import time
from Incremental_learning.NN_modded import *
from Incremental_learning.Model import *
from Incremental_learning.NN import *
from Incremental_learning.ConfusionMatrix import *
from Descriptors.Compute_Descriptors import *
from Obj_segment import *
from Scene import *
import sys
import cPickle
import gzip
import math

Time_ini = time.time()
d = sys.argv[1]
if d == "SIFT":
    mode = 'points'
else:
    mode = 'hist'
limit = int(sys.argv[2])
if sys.argv[2] == '20':
    mini = "Limited"
else:
    mini = "No_Limit"
if len(sys.argv)==4:
    Clase = sys.argv[3]
else:
    Clase = None
path = '/media/iglu/Data/Data2'

def set_name(name):
    if name == 'Cereal' or name.title() == 'Cerealbox':
        return 'CerealBox'
    elif name == 'Tea' or name.title() == 'Teabox':
        return 'TeaBox'
    elif name == 'Big' or name.title() == 'Bigmug':
        return 'BigMug'
    elif name == 'Water' or name.title() == 'Waterbottle':
        return 'WaterBottle'
    elif name == 'Diet' or name.title() == 'Dietcoke':
        return 'DietCoke'
    else:
        return name
def train(NN_BoW,candidate,FM,L1,L2):
    if d in candidate.Descriptors.keys():
        M = Model(mode, candidate.Descriptors[d], None,candidate.patch, (FM.user,FM.action))
        out = NN_BoW.train(M, L1.rstrip('\n').title())
    if candidate.Descriptors_T is not None and d in candidate.Descriptors_T.keys():
        M = Model(mode, candidate.Descriptors_T[d], None,candidate.patch_T, (FM.user,FM.action))
        out = NN_BoW.train(M, L2.rstrip('\n').title())

def test(NN_BoW,candidate,M_BoW,FM):
    L1 = None
    L2 = None
    # M = Model(mode, candidate.Descriptors[d], None, None)
    # out = NN_BoW.ntest(M, 3)
    # out = [set_name(o) for o in out]
    # if len(out) <3:
    #     print out
    # if out[2] == out[1]:
    #     label = out[1]
    # else:
    #     label = out[0]
    # L1 = label
    # if candidate.Label is not None:
    #     M_BoW.add_pair(label.title(), set_name(candidate.Label.rstrip('\n')).title())
    # else:
    #     M_BoW.add_pair(label.title(), set_name(FM.Speech[1].rstrip('\n')).title())
    # if candidate.Descriptors_T is not None and d in candidate.Descriptors_T.keys():
    #     M = Model(mode, candidate.Descriptors_T[d], None, None)
    #     out = NN_BoW.ntest(M, 3)
    #     out = [set_name(o) for o in out]
    #     if out[2] == out[1]:
    #         label = out[1]
    #     else:
    #         label = out[0]
    #     L2 = label
    #     if candidate.Label is not None:
        #     M_BoW.add_pair(label.title(), set_name(candidate.Label.rstrip('\n')).title())
        # else:
        #     M_BoW.add_pair(label.title(), set_name(FM.Speech[1].rstrip('\n')).title())
    return L1,L2
i = 0
users = ["user"+n.__str__() for n in xrange(1,11)]
actions = ['point_1', 'point_2', 'point_3', 'point_4', 'point_5', 'point_6', 'point_7', 'point_8', 'point_9', 'point_10','show_1',  'show_2',  'show_3',  'show_4',  'show_5',  'show_6',  'show_7',  'show_8',  'show_9',  'show_10']
# actions = ['show_1',  'show_2',  'show_3',  'show_4',  'show_5',  'show_6',  'show_7',  'show_8',  'show_9',  'show_10']
actions2 = ['talk_1']
# Lista = [(u,v) for u in users for v in actions]
# Lista2 = [(u,v) for u in users for v in actions2]
# np.random.shuffle(Lista)
# np.random.shuffle(Lista2)
List = []
V = []

NN_BoW = NN_new(0.1, limit)
for user in users:
# for (user, action) in Lista:
    for action in actions:
        files = gzip.open(path + "/" + user + "_" + action + ".gpz", 'r')
        FM = cPickle.load(files)
        files.close()
        if FM.Candidate_patch is None:
            print user + " " + action + " Has no Candidate"
            continue
        # if i < 5:
        for candidate in FM.Candidate_patch:
            if candidate.Label is not None:
                train(NN_BoW,candidate,FM,candidate.Label,candidate.Label)
NN_BoW.visualization("/home/iglu/catkin_ws/src/MIL/Outputs/System_data")
    # else:
    #
    #         if candidate.Label is not None:
    #             L1,L2 = test(NN_BoW,candidate,M_BoW,FM)
    #             if set_name(candidate.Label.rstrip('\n').title()) not in NN_BoW.labels:
    #                 V.append(1)
    #             else:
    #                 V.append(0)
    #     for candidate in FM.Candidate_patch:
    #         if candidate.Label is not None:
    #             train(NN_BoW, candidate,FM,L1,L2)


    # print user
    # M_BoW = ConfusionMatrix()
    # for action in actions2:
    #     files = gzip.open(path + "/" + user + "_" + action + ".gpz", 'r')
    #     FM = cPickle.load(files)
    #     files.close()
    #     for candidate in FM.Objects:
    #         L1, L2 = test(NN_BoW, candidate, M_BoW, FM)
    #         if L1 in candidate.Label:
    #             M_BoW.add_pair(L1,L1)
    #         else:
    #             M_BoW.add_pair(L1, candidate.Label[0])
    #         if L2 in candidate.Label:
    #             M_BoW.add_pair(L2,L2)
    #         else:
    #             M_BoW.add_pair(L2, candidate.Label[0])
    # v = M_BoW.show_info("Prueba_SS.jpg", False)
    # print v

                # cv2.imwrite("Outputs/Test_Final/" + i.__str__() + "/Candidate_" + N.__str__() + "_" + L1 + ".jpg",
                #             candidate.patch)
                # N += 1
                # cv2.imwrite("Outputs/Test_Final/" + i.__str__() + "/Candidate_" + N.__str__() + "_" + L2 + ".jpg",
                #             candidate.patch_T)
                # N += 1
            # v = M_BoW.show_info("", False)
            # List.append((v,np.unique(V,return_counts=True)))
            # NN_BoW.dump("Outputs/Ste/NN_"+i.__str__()+"_part.pkl")
i+=1
# N=0
# M_BoW = ConfusionMatrix()
# for (user, action) in Lista2:
#     files = gzip.open(path + "/" + user + "_" + action + ".gpz", 'r')
#     FM = cPickle.load(files)
#     files.close()
#     for candidate in FM.Objects:
#         L1, L2 = test(NN_BoW, candidate, M_BoW, FM)
#         if L1 in candidate.Label:
#             M_BoW.add_pair(L1, L1)
#         else:
#             M_BoW.add_pair(L1, candidate.Label[0])
#         if L2 in candidate.Label:
#             M_BoW.add_pair(L2, L2)
#         else:
#             M_BoW.add_pair(L2, candidate.Label[0])
# v = M_BoW.show_info("Prueba_SS.jpg", True)
# List.append((v,np.unique(V,return_counts=True)))
# NN_BoW.dump("Outputs/Ste/NN_"+i.__str__()+"_part.pkl")
# for L in List:
#     print L[0].__str__()+";"+L[1]
