import cv2
import numpy as np
from Interaction import *
import sys
import time
from Scene.Scene import *
from Scene.FMI import *
from Obj_segment import *
from Descriptors.Compute_Descriptors import *
from Obj_segment.watershed import Sp_Water
import cPickle
import gzip
import subprocess
import json
import os
from Config import *
from Incremental_learning.NN import *
from Incremental_learning.Model import *

Time_ini = time.time()
arg = sys.argv

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

user = arg[1]
action = arg[2]
existe = False
if os.path.exists(path + "/" + user + "/" + action):
    print user
    print action
    Incremental = NN(0.1, 20)
    Incremental.load(Datas)
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
    ans = raw_input("Test or Training?: ")
    if ans.capitalize() == 'Training':
        I_front = Interaction_Multiple.Mutiple_Interaction()
        if FM.Speech[0] != 'The...':
            I_front.Calculate_Interaction(FM)
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
                    print candidate.Label
                    candidate.Descriptors=D.calculate_D(candidate.patch,candidate.patch_d,["HC"])
                    if candidate.patch_T is not None:
                        candidate.Descriptors_T = D.calculate_D(candidate.patch_T, candidate.patch_d, ["HC"])

        else:
            FM.Values["Class_N"]='Speak'
            T = Obj_candidate.get_candidate(FM)
            for candidate in T:
                if candidate.patch is None:
                    continue
                candidate.Descriptors = D.calculate_D(candidate.patch, candidate.patch_d, ["HC"])
                if candidate.patch_T is not None:
                    candidate.Descriptors_T = D.calculate_D(candidate.patch_T, candidate.patch_d,
                                                            ["HC"])
            for S in FM.Images:
                S.addObj(T)
            FM.Objects = T
            FM.Values["Learning_Model"] = "/home/iglu/catkin_ws/src/MIL/Outputs/Manual/Limited/Train_w_HC_"+user+"_manual.pkl"
            Search_Reference.Search_Ref(FM)
            for candidate in FM.Candidate_patch:
                candidate.Descriptors = D.calculate_D(candidate.patch, candidate.patch_d, ["HC"])
                if candidate.patch_T is not None:
                    candidate.Descriptors_T = D.calculate_D(candidate.patch_T, candidate.patch_d,
                                                            ["HC"])
        for candidate in FM.Candidate_patch:
            M = Model('hist', candidate.Descriptors["HC"], None, candidate.patch, None)
            Incremental.train(M, candidate.Label)
        Incremental.dump(Datas)
        print "Saving..."
        files = gzip.open(path_output+ "/"+user + "_" + action + ".gpz", 'w')
        cPickle.dump(FM, files, -1)
        files.close()
    else:
        i = 0
        FM.Objects = Obj_candidate.get_candidate(FM)
        if FM.Objects is None:
            print "Error, No Object Found"
            exit()
        for candidate in FM.Objects:
            candidate.Descriptors = D.calculate_D(candidate.patch, candidate.patch_d, ["HC"])
            if candidate.patch_T is not None:
                candidate.Descriptors_T = D.calculate_D(candidate.patch_T, candidate.patch_d,
                                                        ["HC"])
            M = Model('hist', candidate.Descriptors["HC"], None,candidate.patch, None)
            out = Incremental.ntest(M,3)
            out = [set_name(o) for o in out]
            if len(out) < 3:
                label = out[0]
            elif out[2] == out[1]:
                label = out[1]
            else:
                label = out[0]
            cv2.imwrite(path_output+ "/Candidate_"+user + "_" + action + "_"+label+ "_"+ i.__str__()+".jpg",candidate.patch)
            i+=1
            if candidate.patch_T is not None:
                candidate.Descriptors_T = D.calculate_D(candidate.patch_T, candidate.patch_d,
                                                        ["HC"])
                M = Model('hist', candidate.Descriptors_T["HC"], None, candidate.patch_T, None)
                out = Incremental.ntest(M, 3)
                out = [set_name(o) for o in out]
                if len(out) < 3:
                    label = out[0]
                elif out[2] == out[1]:
                    label = out[1]
                else:
                    label = out[0]
                cv2.imwrite(
                    path_output + "/Candidate_" + user + "_" + action + "_" + label + "_" + i.__str__() + ".jpg",
                    candidate.patch_T)
                i += 1
        print "All objects processed..."
