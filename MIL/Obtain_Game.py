import cv2
import numpy as np
import sys
import time
from Scene.Scene import *
from Scene.FMI import *
from Obj_segment import *
from Descriptors.Compute_Descriptors import *
from Incremental_learning.NN import *
from Obj_segment import Maskrcnn
import cPickle
import gzip
import subprocess
import json
import os


def get_dim( Img, Mask1):
    # Process the Mask of the top camera
    Mask_1 = Mask1.copy()
    kernel = np.ones((7, 7), np.uint8)
    Mask_1 = cv2.dilate(Mask_1, kernel, 1)
    kernel = np.ones((4, 4), np.uint8)
    Mask_1 = cv2.erode(Mask_1, kernel, 1)
    edged = cv2.Canny(Mask_1, 1, 240)

    # Obtain the total of pixels in the mask
    total = sum(sum(Mask_1[:, :, 0]))

    # Find the biggest countour (The table)
    (im, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_cc = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(max_cc) < 0.6 * total:
        return None, None

    # Create a bounding box for the Table and rotate the images so the angle is 0
    x, y, w, h = cv2.boundingRect(max_cc)
    dim = (x, y, w, h)
    img = Img[y:y + h, x:x + w]
    return dim, img

Time_ini = time.time()
arg = sys.argv
id_game = 0
path_img = '/home/iglu/GWhat/guesswhat/new_data/img/raw/'
path = "/media/iglu/Data/DatasetIglu"
paths = '/media/iglu/Data/Data2/'
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
all = []
mask = Maskrcnn.maskrcnn(False)
actions = ['point_1', 'point_2', 'point_3', 'point_4', 'point_5', 'point_6', 'point_7', 'point_8', 'point_9', 'point_10','show_1',  'show_2',  'show_3',  'show_4',  'show_5',  'show_6',  'show_7',  'show_8',  'show_9',  'show_10', 'talk_1','talk_2']
users = ["user" + n.__str__() for n in xrange(1, 11)]
for user in users:
    for action in actions:
        if os.path.exists(path + "/" + user + "/" + action):
            print user
            print action
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
                RGB_top = cv2.imread(path + "/" + user + "/" + action + "/k2" + "/RGB/" + file1_t)
                # Depth_top = np.load(path + "/" + user + "/" + action + "/k2" + "/Depth/" + file2_t)
                Mask_top = cv2.imread(path + "/" + user + "/" + action + "/k2" + "/MTA/" + file1_t)
                break
            dim,img = get_dim(RGB_top,Mask_top)
            if img is None:
                continue
            sample = {}
            cv2.imwrite( path_img+id_game.__str__()+".jpg",img)
            cv2.imwrite( path_img+"valid/"+id_game.__str__()+".jpg",img)            
            cv2.imwrite( path_img+"test/"+id_game.__str__()+".jpg",img)
            qas = []
            l, kp = mask.maskrcnn(img)
            seg = []
            i = 0
            for obj in l:
                if obj[2] * obj[3] > 0.8 * img.shape[0] * img.shape[1]:
                    continue
                i += 1
                cat = obj[4]
                current_object = {
                    "category_id": cat,
                    "bbox": np.array(obj).tolist(),
                    "category": class_names[cat],
                    "segment": [],
                    "id": i,
                    "area": 0
                }
                seg.append(current_object)
            sample["id"] = id_game
            sample["qas"] = qas
            sample["image"] = {
                "id": id_game,
                "width": dim[2],
                "height": dim[3],
                "file_name": path_img+id_game.__str__()+".jpg"
            }

            sample["objects"] = seg
            for objex in seg:
                sample["object_id"] = objex["id"]
                sample["guess_object_id"] = objex["id"]
                sample["status"] = "failure"
                id_game +=1

                all.append(sample.copy())
import random
random.shuffle(all)
slize = len(all)/10
train = all[0:slize*8]
valid = all[slize*8:slize*9]
test = all[slize*9:]
with gzip.open('/home/iglu/GWhat/guesswhat/new_data/MIL.train.jsonl.gz', 'wb') as f:
    for sample in train:
        f.write(str(json.dumps(sample)).encode())
        f.write(b'\n')
with gzip.open('/home/iglu/GWhat/guesswhat/new_data/MIL.valid.jsonl.gz', 'wb') as f:
    for sample in valid:
        f.write(str(json.dumps(sample)).encode())
        f.write(b'\n')
with gzip.open('/home/iglu/GWhat/guesswhat/new_data/MIL.test.jsonl.gz', 'wb') as f:
    for sample in test:
        f.write(str(json.dumps(sample)).encode())
        f.write(b'\n')
