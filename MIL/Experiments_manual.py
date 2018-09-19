import sys
import os
import timeit
from Incremental_learning.Model import *
from Incremental_learning.NN_modded import *
from Incremental_learning.ConfusionMatrix import *
from Offline_Learning.SVM import *
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

d = sys.argv[1]
if d == 'SIFT':
    Mode = 'points'
else:
    Mode = 'hist'
limit = int(sys.argv[2])
if sys.argv[2] == '20':
    mini = "Limited"
else:
    mini = "No_Limit"
Results = []
path = '/home/pazagra/Data2/Manual_Data'
for i in xrange(1,11):
    List = []
    Data = []
    Label = []
    # NN_BoW = NN_new(0.2,20)
    NN_BoW = SVM_offline("SVM_inc.pkl",100.0,0.001)
    users = ["user"+n.__str__() for n in xrange(1,11) if n != i]
    user_c = "user"+i.__str__()
    print user_c+"_Exp"
    # if os.path.exists("/home/iglu/catkin_ws/src/MIL/Outputs/Manual/"+"Train_wo_FC7_"+user_c+"_manual.pkl"):
    #     continue
    for user in users:
        # print user_c + "_Train_"+user
        files = gzip.open(path+"/" + user + ".gpz", 'r')
        FM = cPickle.load(files)
        files.close()
        if FM.Candidate_patch is None:
            print user+" Has no Candidate"
            continue
        for candidate in FM.Candidate_patch:
            if candidate.Label is not None:
                if d in candidate.Descriptors.keys():
                    # M = Model.Model(Mode, candidate.Descriptors[d], None, None)
                    Data.append(candidate.Descriptors[d])
                    Label.append(candidate.Label.rstrip('\n'))
                    # out = NN_BoW.train(candidate.Descriptors[d], candidate.Label.rstrip('\n'))
                    # out = NN_BoW.train(M, candidate.Label.rstrip('\n'))
        # print Label
        NN_BoW.train(np.array(Data),np.array(Label))
        # v = NN_BoW.visualization()
        # print user_c + "_Test_"+user
        # image = NN_BoW.user_hist()
        # cv2.imwrite("/home/iglu/catkin_ws/src/MIL/Outputs/Manual/"+mini+"/Hist_"+user_c+"_"+d+"_"+user+"_manual.jpg",image)
        M_BoW = ConfusionMatrix()
        files = gzip.open(path + "/" + user_c + ".gpz", 'r')
        FM = cPickle.load(files)
        files.close()
        if FM.Candidate_patch is None:
            print user_c  + " Has no Candidate"
            continue
        for candidate in FM.Candidate_patch:
            if d in candidate.Descriptors.keys():
                # M = Model.Model(Mode, candidate.Descriptors[d], None, None)
                # M = Model('hist', candidate.Descriptors[d], None, None)
                # out = NN_BoW.ntest(M, 3)
                # out = NN_BoW.ntest(candidate.Descriptors[d], 3)
                out = NN_BoW.test(candidate.Descriptors[d])[0]
                # print out
                # print candidate.Label
                # print out
                # out = [set_name(o) for o in out]
                # if out[2] == out[1]:
                #     label = out[1]
                # else:
                #     label = out[0]
                # if candidate.Label is not None:
                    # print candidate.Label.rstrip('\n')
                    # print label.title()
                    # M_BoW.add_pair(out.title(), set_name(candidate.Label.rstrip('\n')).title())
                # else:
                    # M_BoW.add_pair(out.title(), set_name(FM.Speech[1].rstrip('\n')).title())
                # out = NN_BoW.test(candidate.Descriptors[d])
                # if candidate.Label.rstrip in out:
                #     label = candidate.Label
                # else:
                # print out
                # print candidate.Label
                # print"*********"
                label = out
                if label == "":
                    continue
                else:
                    if candidate.Label is not None:
                        M_BoW.add_pair(label, candidate.Label.rstrip('\n'))
                    else:
                        M_BoW.add_pair(label, FM.Speech[1].rstrip('\n'))
        M_BoW.save_confusion("/home/pazagra/Output/"+user+"_manual.npy")
        v = M_BoW.show_info("",False)
        List.append(v)
    Results.append(List)
print Results
    # NN_BoW.dump("/home/iglu/catkin_ws/src/MIL/Outputs/Manual/"+mini+"/Train_w_"+d+"_"+user_c+"_manual.pkl")
