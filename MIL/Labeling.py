import json
import os
import cv2
import Interaction.Caffe2
import numpy as np
import time
path = "/media/iglu/Data/DatasetIglu"
users = ["user"+n.__str__() for n in xrange(1,11)]
actions = ['point_1', 'point_2', 'point_3', 'point_4', 'point_5', 'point_6', 'point_7', 'point_8', 'point_9', 'point_10','show_1',  'show_2',  'show_3',  'show_4',  'show_5',  'show_6',  'show_7',  'show_8',  'show_9',  'show_10']
not_found = 0
data = {}
count = {}
Skol = Interaction.Caffe2.skeleton("")
for user in users:
    for action in actions:
        if os.path.exists(path + "/" + user + "/" + action):
            print user
            print action
            f_speech = open(path + "/" + user + "/" + action + "/speech.txt",'r')
            f_front = open(path + "/" + user + "/" + action + "/k1" + "/List.txt", 'r')
            f_top = open(path + "/" + user + "/" + action + "/k2" + "/List.txt", 'r')
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

                if os.path.exists(path + "/" + user + "/" + action + "/k1" + "/skeleton/"+file1.replace("jpg","json")):
                    skel = open(path + "/" + user + "/" + action + "/k1" + "/skeleton/"+file1.replace("jpg","json"), 'r')
                    skeleton = json.load(skel)
                    t = time.time()
                    canvas, Skil = Skol.get_skeleton(RGB_front,[0.7])
                    print time.time()-t
                    for key in skeleton.keys():
                        a = skeleton[key]
                        if key not in Skil.keys():
                            not_found+=1
                            continue
                        b = Skil[key]
                        if key not in data.keys():
                            data[key] = np.linalg.norm(np.array(a)-np.array(b))
                            count[key] = 1
                        else:
                            data[key] += np.linalg.norm(np.array(a)-np.array(b))
                            count[key] += 1
sum =0
countt = 0
for key in data.keys():
    print key
    sum+= data[key]
    countt+=count[key]
    print data[key]/count[key]

print "TOTAL"
print sum/countt

print "NOT"
print not_found