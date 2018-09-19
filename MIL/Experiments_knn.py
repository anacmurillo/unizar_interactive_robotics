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
import sys
import cPickle
import gzip
import math


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
actions = ['point_1', 'point_2', 'point_3', 'point_4', 'point_5', 'point_6', 'point_7', 'point_8', 'point_9', 'point_10','show_1',  'show_2',  'show_3',  'show_4',  'show_5',  'show_6',  'show_7',  'show_8',  'show_9',  'show_10', 'talk_1','talk_2']
# actions = ['point_1', 'point_2', 'point_3', 'point_4', 'point_5', 'point_6', 'point_7', 'point_8', 'point_9', 'point_10','show_1',  'show_2',  'show_3',  'show_4',  'show_5',  'show_6',  'show_7',  'show_8',  'show_9',  'show_10']
if Clase is not None:
    actions = [ac for ac in actions if ac.startswith(Clase)]

for i in xrange(1,11):
    List = []
    NN_BoW = NN_new(0.1, limit)
    # NN_BoW = NN(0.1, limit, limit)
    users = ["user"+n.__str__() for n in xrange(1,11) if n != i]
    user_c = "user"+i.__str__()
    print user_c+"_Exp"
    # if os.path.exists("/home/iglu/catkin_ws/src/MIL/Outputs/Automatic/"+mini+"/Train_wo_"+d+"_"+user_c+".pkl"):
    #     continue
    for user in users:
        # print user_c + "_Train_"+user
        for action in actions:
            files = gzip.open(path+"/" + user + "_" + action + ".gpz", 'r')
            FM = cPickle.load(files)
            files.close()
            if FM.Candidate_patch is None:
                print user+" "+action+" Has no Candidate"
                continue
            for candidate in FM.Candidate_patch:
                if candidate.Label is not None:
                    if d in candidate.Descriptors.keys() :
                        M = Model(mode, candidate.Descriptors[d], None, None)
                        out = NN_BoW.train(M, candidate.Label.rstrip('\n').title())
                        # out = NN_BoW.train((candidate.Descriptors[d],FM.user), set_name(candidate.Label.rstrip('\n')))
                        if candidate.Descriptors_T is not None and d in candidate.Descriptors_T.keys():
                            M = Model(mode, candidate.Descriptors_T[d], None, None)
                            out = NN_BoW.train(M, candidate.Label.rstrip('\n').title())
                            # out = NN_BoW.train((candidate.Descriptors_T[d], FM.user), set_name(candidate.Label.rstrip('\n')))
        # print user_c + "_Test_"+user
        # image = NN_BoW.user_hist()
        # cv2.imwrite(
        #     "/home/iglu/catkin_ws/src/MIL/Outputs/Automatic/" + mini + "/Hist_" + user_c + "_" + d + "_" + user + "_manual.jpg",
        #     image)
        M_BoW = ConfusionMatrix()
        for action in actions:
            files = gzip.open(path + "/" + user_c + "_" + action + ".gpz", 'r')
            FM = cPickle.load(files)
            files.close()
            if FM.Candidate_patch is None:
                # print user_c + " " + action + " Has no Candidate"
                continue
            for candidate in FM.Candidate_patch:
                if d in candidate.Descriptors.keys():
                    # M = Model(mode, candidate.Descriptors[d], None, None)
                    # n, label, D = NN_BoW.test(M)
                    # n, label, D = NN_BoW.test(candidate.Descriptors[d])
                    M = Model('hist', candidate.Descriptors[d], None, None)
                    out = NN_BoW.ntest(M, 3)
                    # out = NN_BoW.ntest(candidate.Descriptors[d], 3)
                    out = [set_name(o) for o in out]
                    if out[2] == out[1]:
                        label = out[1]
                    else:
                        label = out[0]
                    if candidate.Label is not None:
                        # print candidate.Label.rstrip('\n')
                        # print label.title()
                        M_BoW.add_pair(label.title(), set_name(candidate.Label.rstrip('\n')).title())
                    else:
                        M_BoW.add_pair(label.title(), set_name(FM.Speech[1].rstrip('\n')).title())
                    if candidate.Descriptors_T is not None and d in candidate.Descriptors_T.keys():
                        # M = Model(mode, candidate.Descriptors_T[d], None, None)
                        # n, label, D = NN_BoW.test(M)
                        # n, label, D = NN_BoW.test(candidate.Descriptors[d])
                        M = Model('hist', candidate.Descriptors[d], None, None)
                        out = NN_BoW.ntest(M, 3)
                        # out = NN_BoW.ntest(candidate.Descriptors[d], 3)
                        out = [set_name(o) for o in out]
                        if out[2] == out[1]:
                            label = out[1]
                        else:
                            label = out[0]
                        if candidate.Label is not None:
                            M_BoW.add_pair(label.title(), set_name(candidate.Label.rstrip('\n')).title())
                        else:
                            M_BoW.add_pair(label.title(), set_name(FM.Speech[1].rstrip('\n')).title())
        M_BoW.save_confusion("/home/iglu/catkin_ws/src/MIL/Outputs/Automatic/"+mini+"/" + user_c + "_MC"+d+"_"+user)
        v = M_BoW.show_info("",False)
        List.append(v)
    NN_BoW.dump("/home/iglu/catkin_ws/src/MIL/Outputs/Automatic/"+mini+"/Train_wo_"+d+"_"+user_c+".pkl")
    print List
    # T2 = time.time() - Time_ini
    # print T2