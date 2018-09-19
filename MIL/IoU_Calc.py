import os
import cv2
import json
from Obj_segment import IoU,Obj_Cand,Rect
from Scene import *
import numpy as np

path = "/home/pazagra/Dataset/ManualObjectLabel/"
path_mask = "/home/pazagra/Dataset/ManualObjectLabel/mask/"
users = ["user"+i.__str__()+".jpg" for i in xrange(0,11)]
users_j = ["user"+i.__str__()+".json" for i in xrange(0,11)]
users_m = ["user"+i.__str__()+"_m.jpg" for i in xrange(0,11)]
Obj_candidate = Obj_Cand.Object_Cand()
Total_iou= []
for n in xrange(1,11):
    print users[n]
    img = cv2.imread(path+users[n])
    # print img.shape
    # mask = cv2.imread(path_mask+users_m[n])
    # print mask.shape
    file = open(path + users_j[n], 'r')
    Manual_seg = json.load(file)
    FM = FMI.Full_Movie(users[n],'','')
    files = [path+users[n], None, path_mask + users_m[n], path + users[n],None, path_mask + users_m[n]]
    Scen = Scene.Scene(files[0], files[1], files[2], files[3], files[4], files[5], files)
    S=[Scen]
    FM.Images = S
    Objects = Obj_candidate.get_candidate(FM)
    # print Objects
    M_Obj= []
    for o in xrange(len(Manual_seg['shapes'])):
        Obj = Manual_seg['shapes'][o]
        bbox = Obj['points']
        label = Obj["label"]
        dim0 =[p[0] for p in bbox]
        dim1 = [p[1] for p in bbox]
        p1 = Rect.Point(min(dim0),min(dim1))
        p2 = Rect.Point(max(dim0), max(dim1))
        rect = Rect.Rect(p1,p2)
        M_Obj.append((rect,label))
    for obj in Objects:
        p1,p2 = obj.BB_top.two_point()
        BB = [p1[0],p1[1],p2[0],p2[1]]
        maax= 0
        for man in M_Obj:
            p1, p2 = man[0].two_point()
            BB2 = [p1[0], p1[1], p2[0], p2[1]]
            value = IoU.iou(BB,BB2)
            if value > maax:
                maax = value
        if maax > 0.3:
            Total_iou.append(maax)
print np.mean(np.array(Total_iou))
print len(Total_iou)

