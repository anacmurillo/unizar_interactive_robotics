import cv2
import numpy as np
from Interaction import *
import sys
import time
from Scene.Scene import *
from Scene.FMI import *
from Obj_segment import *
from Descriptors.Compute_Descriptors import *
from Incremental_learning.NN import *
from Obj_segment.watershed import Sp_Water
import cPickle
import gzip
import subprocess
import json
import os

Time_ini = time.time()
arg = sys.argv

path = "/home/pazagra/Dataset"
paths = '/home/pazagra/Data/'
user = arg[1]
action = arg[2]
existe = False
if os.path.exists(path + "/" + user + "/" + action):
    print user
    print action
    Incremental = NN(0.1, 20, 20)
    Incremental.load("RR.pkl")
    I_front = Interaction_Multiple.Mutiple_Interaction()
    Obj_candidate = Obj_Cand.Object_Cand()
    D = D_calculator()
    f_speech = open(path + "/" + user + "/" + action + "/speech.txt",'r')
    f_front = open(path + "/" + user + "/" + action + "/k1" + "/List.txt", 'r')
    f_top = open(path + "/" + user + "/" + action + "/k2" + "/List.txt", 'r')
    speech = f_speech.readlines()

    FM = Full_Movie(user,action,speech[0])
    if os.path.exists(path + "/" + user + "/" + action + "/k2" + "/Speech.txt"):
        Sspeak = open(path + "/" + user + "/" + action + "/k2" + "/Speech.txt",'r')
        lines = Sspeak.readlines()
        lines = [line.rstrip('\n').split(' ') for line in lines]
        Sspeak.close()
        FM.Values["Speak"] = lines
    Data = []
    d_f = f_front.readlines()
    d_t = f_top.readlines()
    n = min(len(d_f),len(d_t))/4
    for i in xrange(n):
        num = i*4
        Time = d_f[num]
        file1 = d_f[num+1].rstrip('\n')
        file2 = d_f[num+2].rstrip('\n')
        Label = d_f[num+3].rstrip('\n')
        Time_t = d_t[num]
        file1_t = d_t[num+1].rstrip('\n')
        file2_t = d_t[num+2].rstrip('\n')
        Label_t = d_t[num+3].rstrip('\n')

        # RGB_front = cv2.imread(path + "/" + user + "/" + action + "/k1" + "/RGB/"+file1)
        # Depth_front = np.load(path + "/" + user + "/" + action + "/k1" + "/Depth/"+file2)
        # Mask_front = cv2.imread(path + "/" + user + "/" + action + "/k1" + "/MTA/"+file1)
        # RGB_top = cv2.imread(path + "/" + user + "/" + action + "/k2" + "/RGB/" + file1_t)
        # Depth_top = np.load(path + "/" + user + "/" + action + "/k2" + "/Depth/" + file2_t)
        # Mask_top = cv2.imread(path + "/" + user + "/" + action + "/k2" + "/MTA/" + file1_t)
        files = [path + "/" + user + "/" + action + "/k1" + "/RGB/"+file1,path + "/" + user + "/" + action + "/k1" + "/Depth/"+file2,path + "/" + user + "/" + action + "/k1" + "/MTA/"+file1,
        path + "/" + user + "/" + action + "/k2" + "/RGB/" + file1_t,path + "/" + user + "/" + action + "/k2" + "/Depth/" + file2_t,path + "/" + user + "/" + action + "/k2" + "/MTA/" + file1_t]
        Scen = Scene(files[0],files[1],files[2],files[3],files[4],files[5],files)
        Scen.Values["Interaction_GT"] = Label.replace('\n','')

        if os.path.exists(path + "/" + user + "/" + action + "/k1" + "/skeleton/"+file1.replace("jpg","json")):
            skel = open(path + "/" + user + "/" + action + "/k1" + "/skeleton/"+file1.replace("jpg","json"), 'r')
            skeleton = json.load(skel)
            Scen.Skeleton=skeleton
        Data.append(Scen)
    FM.Images = Data
    # num=0
    # Cand=[]
    # for S in FM.Images:
    #     dim,img = Obj_candidate.get_dim(cv2.imread(S.RGB_top),S.Mask_top)
    #     if dim is None or img is None:
    #         continue
    #     S.Values["dim"]=dim
    #     if num <= 5:
    #         num += 1
    #         cv2.imwrite("/home/iglu/tfenv/maskrcnn/Mask_RCNN/Test.jpg", img)
    #         subprocess.Popen(["bash", "/home/iglu/catkin_ws/src/MIL/Mask.sh"]).wait()
    #         fi = open("Output.txt", 'r')
    #         for l in fi.readlines():
    #             n = l.rstrip("\n")[1:-1]
    #             n = n.split(",")
    #             n = [int(ll) for ll in n]
    #             r1 = dim[1] + n[0]
    #             r2 = dim[1] + n[1]
    #             c1 = dim[0] + n[2]
    #             c2 = dim[0] + n[3]
    #             C = Obj_candidate.top_to_front(c1, c2, r1, r2, cv2.imread(S.RGB_front))
    #             Cand.append(C)
    #         fi.close()
    #         os.remove("Output.txt")
    #         T,img = Sp_Water(cv2.imread(S.RGB_front),img,dim)
    #         for c in T:
    #             Cand.append(c)
    # NMS.nms(Cand)

    i = 0
    elapsed = 0
    if FM.Speech[0] != 'The...':
        I_front.Calculate_Interaction(FM)
        del I_front
        if FM.Values["Class_N"] in ["Point","Speak"]:
            T = Obj_candidate.get_candidate(FM)
            for S in FM.Images:
                S.addObj(T)
            FM.Objects = T
        else:
            FM.Objects = []
        Search_Reference.Search_Ref(FM)
        for candidate in FM.Candidate_patch:
                candidate.Label = FM.Speech[1].replace('\n','')
                candidate.Descriptors=D.calculate_D(candidate.patch,candidate.patch_d,["HC","ORB","FC7","SIFT"])
                if candidate.patch_T is not None:
                    candidate.Descriptors_T = D.calculate_D(candidate.patch_T, candidate.patch_d, ["HC", "ORB", "FC7","SIFT"])

    else:
        FM.Values["Class_N"]='Speak'
        T = Obj_candidate.get_candidate(FM)
        for candidate in T:
            if candidate.patch is None:
                continue
            candidate.Descriptors = D.calculate_D(candidate.patch, candidate.patch_d, ["HC", "ORB", "FC7", "SIFT"])
            if candidate.patch_T is not None:
                candidate.Descriptors_T = D.calculate_D(candidate.patch_T, candidate.patch_d,
                                                        ["HC", "ORB", "FC7", "SIFT"])
        for S in FM.Images:
            S.addObj(T)
        FM.Objects = T
        FM.Values["Learning_Model"] = "/home/iglu/catkin_ws/src/MIL/Outputs/Manual/Limited/Train_w_HC_"+user+"_manual.pkl"
        Search_Reference.Search_Ref(FM)
        for candidate in FM.Candidate_patch:
            # candidate.Label = FM.Speech[1].replace('\n', '')
            candidate.Descriptors = D.calculate_D(candidate.patch, candidate.patch_d, ["HC", "ORB", "FC7", "SIFT"])
            if candidate.patch_T is not None:
                candidate.Descriptors_T = D.calculate_D(candidate.patch_T, candidate.patch_d,
                                                        ["HC", "ORB", "FC7", "SIFT"])

    print "Saving..."
    files = gzip.open(paths+user + "_" + action + ".gpz", 'w')
    cPickle.dump(FM, files, -1)
    files.close()
else:
    exit()
    existe=True
    files = gzip.open(paths+user + "_" + action + ".gpz", 'r')
    FM= cPickle.load(files)
    files.close()
T2 = time.time() - Time_ini
print T2
exit()
for candidate in FM.Candidate_patch:
    n,label,D = Incremental.test(candidate.Descriptors["HC"],0)
    if n == -1:
        print "I don't know man...I have no knowledge"
        i = Incremental.train((candidate.Descriptors["HC"],FM.user),candidate.Label)
        print "But thanks to you now i know that this is "+candidate.Label+". Thank you."
    elif n == 0:
        print "My Spidersenses tell me that this is "+label+" but i'm not quite sure"
        if label == candidate.Label:
            print "And i was right. Spidersenses never fail."
            i = Incremental.train((candidate.Descriptors["HC"],FM.user), candidate.Label)
        else:
            i = Incremental.train((candidate.Descriptors["HC"],FM.user), candidate.Label)
            if i == 0:
                print "Well, Errare humanum est. En mi defensa dire que no sabia lo que es "+candidate.Label
            elif i == 1:
                print "Well, Errare humanum est. Me lo guardo para el futuro, cuando Skynet conquiste todo."
    else:
        print "I am 99,99% confidence that this is "+label+"."
        if label == candidate.Label:
            print "And i was right,of course."
            i = Incremental.train((candidate.Descriptors["HC"],FM.user), candidate.Label)
        else:
            i = Incremental.train((candidate.Descriptors["HC"],FM.user), candidate.Label)
            if i == 0:
                print "Well, Errare humanum est. En mi defensa dire que no sabia lo que es " + candidate.Label
            elif i == 1:
                print "Well, Errare humanum est. Me lo guardo para el futuro, cuando Skynet conquiste todo."
Incremental.dump("RR.pkl")
files = gzip.open(paths+user+"_"+action+".gpz",'w')
cPickle.dump(FM,files,-1)
files.close()

