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
import cPickle
import gzip

Time_ini = time.time()
arg = sys.argv

I_front = Interaction_Multiple.Mutiple_Interaction()
Obj_candidate = Obj_Cand.Object_Cand()
D = D_calculator()
Incremental = NN(0.1,20,20)
#Incremental.load("Prueba.pkl")

path = "/media/iglu/Data/DatasetIglu"
user = arg[1]
action = arg[2]

f_speech = open(path + "/" + user + "/" + action + "/speech.txt",'r')
f_front = open(path + "/" + user + "/" + action + "/k1" + "/List.txt", 'r')
f_top = open(path + "/" + user + "/" + action + "/k2" + "/List.txt", 'r')
speech = f_speech.readlines()

FM = Full_Movie(user,action,speech[0])
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

    RGB_front = cv2.imread(path + "/" + user + "/" + action + "/k1" + "/RGB/"+file1)
    Depth_front = np.load(path + "/" + user + "/" + action + "/k1" + "/Depth/"+file2)
    Mask_front = cv2.imread(path + "/" + user + "/" + action + "/k1" + "/MTA/"+file1)
    RGB_top = cv2.imread(path + "/" + user + "/" + action + "/k2" + "/RGB/" + file1_t)
    Depth_top = np.load(path + "/" + user + "/" + action + "/k2" + "/Depth/" + file2_t)
    Mask_top = cv2.imread(path + "/" + user + "/" + action + "/k2" + "/MTA/" + file1_t)
    files = [path + "/" + user + "/" + action + "/k1" + "/RGB/"+file1,path + "/" + user + "/" + action + "/k1" + "/Depth/"+file2,path + "/" + user + "/" + action + "/k1" + "/MTA/"+file1,
    path + "/" + user + "/" + action + "/k2" + "/RGB/" + file1_t,path + "/" + user + "/" + action + "/k2" + "/Depth/" + file2_t,path + "/" + user + "/" + action + "/k2" + "/MTA/" + file1_t]

    Scen = Scene(RGB_front,Depth_front,Mask_front,RGB_top,Depth_top,Mask_top,files)
    Scen.Values["Interaction_GT"] = Label
    Data.append(Scen)
FM.Images = Data

T = Obj_candidate.get_candidate(FM)
for S in FM.Images:
    S.addObj(T)
FM.Objects = T

if FM.Speech[0] != 'The':
    I_front.Calculate_Interaction(FM)
    Search_Reference.Search_Ref(FM)
    for candidate in FM.Candidate_patch:
        candidate.Label = FM.Speech[1]
        candidate.Descriptors=D.calculate_D(candidate.patch,None,["HC","ORB"])
else:
    FM.Values["Class_N"]='Speak'
for candidate in FM.Candidate_patch:
    n,label,D = Incremental.test(candidate.Descriptors,0)
    if n == -1:
        print "I don't know man...I have no knowledge"
        i = Incremental.train(candidate.Descriptors,candidate.Label)
        print "But thanks to you now i know that this is "+candidate.Label+". Thank you."
    elif n == 0:
        print "My Spidersenses tell me that this is "+label+" but i'm not quite sure"
        if label == candidate.Label:
            print "And i was right. Spidersenses never fail."
            i = Incremental.train(candidate.Descriptors, candidate.Label)
        else:
            i = Incremental.train(candidate.Descriptors, candidate.Label)
            if i == 0:
                print "Well, Errare humanum est. En mi defensa dire que no sabia lo que es "+candidate.Label
            elif i == 1:
                print "Well, Errare humanum est. Me lo guardo para el futuro, cuando Skynet conquiste todo."
    else:
        print "I am 99,99% confidence that this is "+label+"."
        if label == candidate.Label:
            print "And i was right,of course."
            i = Incremental.train(candidate.Descriptors, candidate.Label)
        else:
            i = Incremental.train(candidate.Descriptors, candidate.Label)
            if i == 0:
                print "Well, Errare humanum est. En mi defensa dire que no sabia lo que es " + candidate.Label
            elif i == 1:
                print "Well, Errare humanum est. Me lo guardo para el futuro, cuando Skynet conquiste todo."
Incremental.dump("RR.pkl")
files = gzip.open(user+"_"+action+".gpz",'w')
cPickle.dump(FM,files,-1)
files.close()
T2 = time.time() - Time_ini
print T2
