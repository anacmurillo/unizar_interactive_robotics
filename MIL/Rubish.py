import cPickle
import gzip
import json
import numpy as np
import cv2
import math
from Interaction import NMS
import json
import Interaction.CNN
import Interaction.Interaction_Recogn
from Obj_segment import Point_Seg, Show_Seg, Candidate, Speak_Seg,watershed,Rect
from Descriptors.Compute_Descriptors import *
import os
from Scene.FMI import *
from scipy.spatial import distance as dist
from skimage import feature
from Obj_segment import Point_Seg


h_p_i = 0
h_s_i = 0
t_p_i = 0
t_s_i = 0
hit_point_v = 0
hit_show_v = 0
total_point_v = 0
total_show_v = 0
path = '/media/iglu/Data/Data2/'
paths = '/media/iglu/Data/DatasetIglu'


def obtain_xy_angle(skeleton):
    def plot_point(point, angle, length):
        # unpack the first point
        x, y = point

        # find the end point
        endy = y + length * math.sin(math.radians(angle))
        endx = x + length * math.cos(math.radians(angle))
        return (int(endx), int(endy))

    if 'hand left' in skeleton.keys() and 'arm left' in skeleton.keys():
        angle = math.atan2(skeleton['hand left'][0] - skeleton['arm left'][0],
                                      skeleton['hand left'][1] - skeleton['arm left'][1])

        start = (skeleton['hand left'][1], skeleton['hand left'][0])
        end = plot_point(start, angle, 200)
        left= (start,end,angle)
    else:
        left = (None,None,None)
    if 'hand right' in skeleton.keys() and 'arm right' in skeleton.keys():
        angle = math.atan2(skeleton['hand right'][0] - skeleton['arm right'][0],
                                      skeleton['hand right'][1] - skeleton['arm right'][1])
        start = (skeleton['hand right'][1], skeleton['hand right'][0])
        end = plot_point(start, angle, 200)
        right = (start, end,angle)
    else:
        right = (None,None,None)
    return left,right

def draw_skeleton(canvas,skel):
    limbSeq = [['chest', 'shoulder right'], ['chest', 'shoulder left'], ['shoulder right', 'arm right'], ['arm right', 'hand right'],['shoulder left', 'arm left'], ['arm left', 'hand left'], ['chest', 'hip right'], ['hip right', 'knee right'], ['knee right', 'foot right'],['chest', 'hip left'], ['hip left', 'knee left'], ['knee left', 'foot left'], ['chest', 'face'], ['face', 'eye right'], ['eye right', 'ear right'], ['face', 'eye left'], ['eye left', 'ear left'], ['shoulder right', 'ear right'], ['shoulder left', 'ear left']]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],		          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],[170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    stickwidth = 4
    for x in range(17):
        index = limbSeq[x]
        if index[0] not in skel.keys() or index[1] not in skel.keys():
            continue
        cur_canvas = canvas.copy()
        Y = [skel[index[0]][1],skel[index[1]][1]]
        X = [skel[index[0]][0],skel[index[1]][0]]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors[x])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas

def new_angle(Scena):
    def PointatD(P_ini, theta, d):
        x2 = P_ini[0] + d * np.sin(theta)
        y2 = P_ini[1] + d * np.cos(theta)
        return (int(y2), int(x2))
    f_depth = np.load(Scena.Depth_front)
    f_depth *= 255.0 / f_depth.max()
    f_depth = np.array(f_depth, np.uint8)
    f_depth = np.dstack((f_depth, f_depth, f_depth))
    gray = cv2.cvtColor(f_depth, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 10, 200)
    kernel = np.ones((4, 4), np.uint8)
    edged = cv2.dilate(edged, kernel, 1)
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.erode(edged, kernel, 1)
    f_depth = np.dstack((edged, edged, edged))
    f_depth = np.array(f_depth, np.uint8)
    if "Hand_Pos" in Scena.Values.keys():
        f_f = f_depth[Scena.Values["Hand_Pos"][0] - 100:Scena.Values["Hand_Pos"][0] + 100,
              Scena.Values["Hand_Pos"][1] - 100:Scena.Values["Hand_Pos"][1] + 100, :].copy()
        f_f[0:1, :, :] = 255
        f_f[-1:, :, :] = 255
        f_f[:, 0:1, :] = 255
        f_f[:, -1:, :] = 255
        l = []
        p_ini = (100, 100)
        while f_f[p_ini[0], p_ini[1], 0] != 0:
            p_ini = (p_ini[0] + 1, p_ini[1] + 1)
        for i in xrange(20,160):
            d = 5
            p2 = PointatD(p_ini, math.radians(i), d)
            while f_f[p2[1], p2[0], 0] != 255:
                d += 0.5
                p2 = PointatD(p_ini, math.radians(i), d)
            if d <= 200 and d > 4 and (p2[0]< 199 and p2[1] < 199 and p2[0]>1 and p2[1]>1):
                cv2.line(f_f,p_ini,p2,(0,255,0),1)
                l.append((i, d, p2))
        if l == []:
            return None,f_depth
        l = sorted(l, key=lambda p: p[1], reverse=True)
        prt = l[0]
        i = prt[0]
        p2 = PointatD(p_ini, math.radians(i), 500)
        cv2.line(f_f, p_ini, p2, (255, 0, 0), 3)
        f_depth[Scena.Values["Hand_Pos"][0] - 100:Scena.Values["Hand_Pos"][0] + 100,
        Scena.Values["Hand_Pos"][1] - 100:Scena.Values["Hand_Pos"][1] + 100, :] = f_f
        return prt[0],f_depth
    else:
        return None,f_depth

def PointatD(P_ini, theta, d):
    x2 = P_ini[0] + d * math.sin(theta)
    y2 = P_ini[1] + d * math.cos(theta)
    return (int(x2), int(y2))

video = cv2.VideoWriter('Outputs/FullDatasetVideo.avi', cv2.cv.CV_FOURCC('D', 'I', 'V', 'X'),
                         25.0, (1280, 480))
users = ["user" + n.__str__() for n in xrange(1, 11)]
# actions = ['talk_1','talk_2']
actions = ['point_1', 'point_2', 'point_3', 'point_4', 'point_5', 'point_6', 'point_7', 'point_8', 'point_9',
           'point_10','show_1',  'show_2',  'show_3',  'show_4',  'show_5',  'show_6',  'show_7',  'show_8',  'show_9',  'show_10','talk_1','talk_2']
# P_seg = Point_Seg.Point_Seg()
i = 0
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
names1 = ['Apple', 'Banana', 'Bowl', 'CerealBox', 'Coke', 'DietCoke', 'Fork', 'Glass', 'Ketchup', 'Kleenex',
               'Knife', 'Lemon', 'Lime', 'Mug', 'Noodles', 'Orange', 'Pringles', 'Plate', 'Spoon', 'BigMug', 'TeaBox',
               'WaterBottle']

pn=0
pt = 0
pv = 0
st=0
ns =0
sv = 0
for user in users:
    print user
    for action in actions:
        print action
        files = gzip.open(path + "/" + user + "_" + action + ".gpz", 'r')
        FM = cPickle.load(files)
        files.close()
        # top = cv2.imread(FM.Images[0].RGB_top)
        for candidate in FM.Candidate_patch:
            cv2.imwrite("Outputs/Candidates/Candidate_" + user + "_" + action +"_" + candidate.Label + "_" + i.__str__() + ".jpg", candidate.patch)
            i+=1
            if candidate.patch_T is not None:
                cv2.imwrite(
                    "Outputs/Candidates/Candidate_" + user + "_" + action + "_" + candidate.Label + "_" + i.__str__() + ".jpg",
                    candidate.patch_T)
                i+=1
            # if set_name(candidate.Label.rstrip('\n').title()) not in names1:
            #     print user
            #     print action
            #     print candidate.Label
            #     print set_name(candidate.Label.rstrip('\n'))
        #     patch = candidate.patch
        #     clas = FM.Values["Class_N"].lower()
        #     Label = candidate.Label
        #     cl = action.split('_')[0]
            # if clas == 'point':
            #     pt +=1
            #     if clas != cl:
            #         pn +=1
            #     else:
            #         pv +=1
            # elif clas == 'show':
            #     st +=1
            #     if clas!= cl:
            #         ns +=1
            #     else:
            #         sv +=1
            # cv2.imwrite("Outputs/Object_Table/Candidate_"+user+"_"+action+"_"+candidate.Values["D"].__str__()+"_"+Label+"_"+i.__str__()+".jpg",patch)
            # i+=1
            # if candidate.patch_T is not None:
                # if clas == 'point':
                #     pt += 1
                #     if clas != cl:
                #         pn += 1
                #     else:
                #         pv += 1
                # elif clas == 'show':
                #     st += 1
                #     if clas != cl:
                #         ns += 1
                #     else:
                #         sv += 1
            # color = (125, 125, 0)
            # p1, p2 = candidate.BB_top.two_point()
            # cv2.rectangle(top, p1, p2, color, 2)
        # for cand, ref in zip(FM.Candidate_patch,FM.Values["Ref"]):
        #     color = (125, 0, 255)
        #     p1, p2 = ref.BB_top.two_point()
        #     cv2.rectangle(top, p1, p2, color, 2)
        #     cv2.putText(top,ref.Label,p2,cv2.FONT_HERSHEY_COMPLEX,1.2,(255,0,0),2)
        #     color = (0, 255, 255)
        #     p1, p2 = cand.BB_top.two_point()
        #     cv2.rectangle(top, p1, p2, color, 2)
        #     cv2.putText(top, cand.Label, p2, cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 0), 2)
        #     patch = cand.patch_T
        #     cv2.imwrite("Outputs/Object_Table/Candidate_" +user+"_"+action+ "_" + i.__str__() + "_" + cand.Label +  "_cand.jpg",patch)
        #     cv2.imwrite(
        #         "Outputs/Object_Table/Candidate_" + user + "_" + action + "_" + i.__str__() + "_" + ref.Label +  "_ref.jpg",
        #         ref.patch_T)
        #     i+=1
        # cv2.imwrite("Outputs/Object_Table/Candidate_" + user + "_" + action + ".jpg", top)

exit()
for user in users:
    print user
    for action in actions:
        print action
        files = gzip.open(path + "/" + user + "_" + action + ".gpz", 'r')
        FM = cPickle.load(files)
        files.close()
        # K = [S.Values["Interaction_recognized"] for S in FM.Images if "Interaction_recognized"  in S.Values.keys()]
        # if len(K) >= 5:
        #     if FM.action.startswith('point'):
        #         if FM.Values["Class_N"] == 'Point':
        #             hit_point_v+=1
        #         total_point_v+=1
        #     else:
        #         if FM.Values["Class_N"] == 'Show':
        #             hit_show_v+=1
        #         total_show_v+=1
        # for c in FM.Objects:
        #     print c.BB_front
        #     cv2.imshow("Im",c.patch)
        #     cv2.waitKey(4000)
        Sp_seg = Speak_Seg.Speak_seg(FM.Values["Learning_Model"])
        Sp_seg.add_candidates(FM.Objects)
        for ob in FM.Objects:
            print ob.Label
            cv2.imshow("Im",ob.patch_T)
            cv2.waitKey(4000)
        for Scena in FM.Images:
            front = cv2.imread(Scena.RGB_front)
            top = cv2.imread(Scena.RGB_top)
            f_depth = np.load(Scena.Depth_front)
            t_depth = np.load(Scena.Depth_top)
            if os.path.exists(Scena.RGB_top.replace("/RGB/","/skeleton/").replace("jpg", "json")):
                 skel = open(Scena.RGB_top.replace("/RGB/","/skeleton/").replace("jpg", "json"), 'r')
                 skeleton = json.load(skel)
            #     continue
            # else:
            #     if not os.path.exists(Scena.RGB_top.replace("/RGB/","/skeleton/").split("Frame")[0]):
            #         os.mkdir(Scena.RGB_top.replace("/RGB/","/skeleton/").split("Frame")[0])
            #     canvas, skeleton = Skeletonn.get_skeleton(top,[0.7])
            #     skl = open(Scena.RGB_top.replace("/RGB/","/skeleton/").replace("jpg", "json"), 'w')
            #     json.dump(skeleton,skl)
            #     skl.close()
            #     top =canvas.copy()
            # Img, dimension = get_images(top,cv2.imread(Scena.Mask_top))
            # Scena.Values["TableDim"] = dimension
            # Candidates, img = watershed.Sp_Water(Img)
            # cv2.imshow("M",img)
            # cv2.waitKey(44)
            # if "hand right" in skeleton.keys():
            #     mano = top[skeleton["hand right"][0]-75:skeleton["hand right"][0]+75,skeleton["hand right"][1]-75:skeleton["hand right"][1]+75]
            #     labels = slic(mano,10,sigma=1)
            #     labels+=1
            #     mano = label2rgb(labels,mano,kind='avg')
            #     top[skeleton["hand right"][0] - 75:skeleton["hand right"][0] + 75,
            #     skeleton["hand right"][1] - 75:skeleton["hand right"][1] + 75] = mano
            # top = draw_skeleton(top.copy(), skeleton)
            # ids = 0
            # Boxes = []
            # for obj in Scena.Objects:
            #     obj.Values["id"]= ids
            #     nnm = obj.BB_front.four_p()
            #     Boxes.append((nnm[0],nnm[1],nnm[3],nnm[4],ids))
            #     ids+=1
            # Boxes = NMS.non_max_suppression_fast(np.array(Boxes),0.5)
            # print len(Boxes)
            # ids = Boxes[:,4]
            # P_seg.add_candidates(FM.Objects)
            # kmeans = MiniBatchKMeans(n_clusters=17).fit(np.array(points_top))
            #
            # for num in xrange(len(kmeans.cluster_centers_)):
            #     indx = np.where( kmeans.labels_ == num)[0]
            #     center = kmeans.cluster_centers_[num]
            #     ind = indx[0]
            #     min_d = np.linalg.norm(center-np.array(Scena.Objects[ind].BB_top.center()))
            #     for nma in xrange(1,len(indx)):
            #         dist = np.linalg.norm(center-np.array(Scena.Objects[indx[nma]].BB_top.center() ))
            #         if dist < min_d:
            #             ind = indx[nma]
            #             min_d = dist
            #     obj = Scena.Objects[ind]
            #     p1, p2 = obj.BB_front.two_point()
            #     cv2.rectangle(front, p1, p2, color, 2)
            #     p1, p2 = obj.BB_top.two_point()
            #     cv2.rectangle(top, p1, p2, color, 2)
            # P_seg.add_candidates(FM.Objects)
            # l, f_d = new_angle(Scena)
            # front = f_d
            if Scena.Objects is not None:
                for obj in Scena.Objects:
                    if obj.patch_d == "MaskRCNN":
                        color = (125,125,0)
                        p1, p2 = obj.BB_front.two_point()
                        cv2.rectangle(front, p1,p2, color, 2)
                        p1, p2 = obj.BB_top.two_point()
                        cv2.rectangle(top, p1,p2,  color, 2)
                    elif obj.patch_d == "WaterShed":
                        color = (0,0,255)
                        p1, p2 = obj.BB_front.two_point()
                        cv2.rectangle(front,p1,p2, color, 2)
                        p1, p2 = obj.BB_top.two_point()
                        cv2.rectangle(top, p1,p2, color, 2)
                    else:
                        color = (255,0,0)
                        p1, p2 = obj.BB_front.two_point()
                        cv2.rectangle(front, p1, p2, color, 2)
                        p1, p2 = obj.BB_top.two_point()
                        cv2.rectangle(top, p1, p2, color, 2)
            L = []
            for speech in FM.Values["Speak"]:
                print speech
                c = Sp_seg.obtain_candidate(speech)
                if c != None:
                    L.append(c)
            FM.Candidate_patch = L

            # if "Hand_Pos" in Scena.Values.keys() and "Hand_Angle" in Scena.Values.keys():
            #     if FM.Values["Class_N"] == 'Point':
            #         for candidate in Scena.Values["Candidates"]:
            #             color = (255, 0, 255)
            #             p1, p2 = candidate.BB_front.two_point()
            #             cv2.rectangle(front, p1, p2, color, 2)
            #             p1, p2 = candidate.BB_top.two_point()
            #             cv2.rectangle(top, p1, p2, color, 2)
            #         p_ini = (Scena.Values["Hand_Pos"][1], Scena.Values["Hand_Pos"][0])
            #         angle = Scena.Values["Hand_Angle"]
            #         cv2.putText(front,(angle).__str__(),p_ini,cv2.FONT_HERSHEY_COMPLEX,2.0,(255,0,0),2)
            #         # print p_ini
            #         if p_ini is not None and angle is not None:
            #             p2 = PointatD(p_ini, math.radians(angle), 370)
            #             cv2.line(front, p_ini, p2, (0, 0, 255), 3)
                        # p2 = PointatD(p_ini, math.radians(0), 370)
                        # cv2.line(front, p_ini, p2, (0, 125, 255), 3)
                        # p2 = PointatD(p_ini, math.radians(90), 370)
                        # cv2.line(front, p_ini, p2, (125, 0, 255), 3)
                        # p2 = PointatD(p_ini, math.radians(180), 370)
                        # cv2.line(front, p_ini, p2, (255, 0, 255), 3)
                        # p2 = PointatD(p_ini, math.radians(270), 370)
                        # cv2.line(front, p_ini, p2, (0, 255, 255), 3)
                        # cv2.circle(front,p_ini,2,(0,255,0),2)
                        # p2 = PointatD(p_ini, math.radians(l), 500)
                        # cv2.line(front, p_ini, p2, (0, 255, 255), 3)
                    #     print "p_seg"
                    # c  = P_seg.obtain_candidate(p_ini, angle,False,front)
                    # print "___________________________"
                    # for cand in c:
                    #     color = (125, 30, 255)
                    #     p1, p2 = cand.BB_front.two_point()
                    #     cv2.rectangle(front, p1,p2, color, 2)
                    #     p1, p2 = cand.BB_top.two_point()
                    #     cv2.rectangle(top, p1, p2, color, 2)
                    # print "____"
                    # angle_left, angle_right = obtain_xy_angle(skeleton)
                    # if Scena.Values["Side"] == 'left':
                    #     angle = angle_left
                    # else:
                    #     angle = angle_right
                    # p_ini= angle[0]
                    # angle = angle[2]
                    # if p_ini is not None and angle is not None:
                    #     p2 = PointatD(p_ini, angle, 200)
                    #     cv2.line(top, p_ini, p2, (0, 125, 0), 3)


                # cv2.circle(front,p1,2,color,2)
                # p1 = obj.BB_top.center()
                # points_top.append(p1)
                # cv2.circle(top,p1,2,color,2)
                # if obj.Values["id"] not in ids:
                #     continue
                # p1, p2 = obj.BB_front.two_point()
                # cv2.rectangle(front, p1, p2, color, 2)
                # p1, p2 = obj.BB_top.two_point()
                # cv2.rectangle(top, p1, p2, color, 2)

            # front = draw_skeleton(front,Scena.Skeleton)
            # if "Candidates" in Scena.Values.keys():
            #     if FM.Values["Class_N"] == 'Point':
            #         for candidate in Scena.Values["Candidates"] :
            #             color = (125, 0, 255)
            #             p1, p2 = candidate.BB_front.two_point()
            #             cv2.rectangle(front, p1,p2, color, 2)
            #             if candidate.patch_T is not None:
            #                 color = (125, 0, 255)
            #                 p1, p2 = candidate.BB_top.two_point()
            #                 cv2.rectangle(top, p1, p2, color, 2)
            #     else:
            # for candidate in FM.Candidate_patch :
            #     color = (0, 125, 255)
            #     p1, p2 = candidate.BB_front.two_point()
            #     cv2.rectangle(front, p1,p2, color, 2)
            #     if candidate.patch_T is not None:
            #         color = (0, 125, 255)
            #         p1, p2 = candidate.BB_top.two_point()
            #         cv2.rectangle(top, p1, p2, color, 2)
                     # color = (0, 125, 255)
                    # p1, p2 = Scena.Values["Candidates"].BB_front.two_point()
                    # cv2.rectangle(front, p1, p2, color, 2)
            # if skeleton is not None:
            #     top = draw_skeleton(top, skeleton)
            I = np.hstack((front, top))
            # print I.shape
            video.write(I)
        # files = gzip.open('/media/iglu/Data/Data2/Manual_Data/' + user + ".gpz", 'w')
        # cPickle.dump(FM, files, -1)
        # files.close()
video.release()
# candidate = Candidate(None,None,None,None,img)
# candidate.add_label(ob.split('.')[0].split('_')[-1])
# candidate.Descriptors = D.calculate_D(candidate.patch, None, ["HC", "ORB", "FC7"])
# list.append(candidate)
# FM.Candidate_patch= list


#         total_show_v+=1
#         if c.Values['Valid']:
#             hit_show_v+=1
#     c = raw_input(c.Label)
#     if c == "y":
#         print "OK"
#     elif c == "n":
#         print "NO"
#     else:
#         print "Partial"
# K = [S.Values["Interaction_recognized"] for S in FM.Images if "Interaction_recognized"  in S.Values.keys()]
# if len(K) < 5:
#     continue
# if FM.action.startswith('point'):
#     K1 = [S.Values["Interaction_recognized"] for S in FM.Images if
#           "Interaction_recognized" in S.Values.keys() and S.Values["Interaction_recognized"] == 'Point']
#     h_p_i += len(K1)
#     t_p_i += len(K)
#     if FM.Values["Class_N"] == 'Point':
#         hit_point_v+=1
#     total_point_v+=1
# else:
#     K1 = [S.Values["Interaction_recognized"] for S in FM.Images if
#           "Interaction_recognized" in S.Values.keys() and S.Values["Interaction_recognized"] == 'Show']
#     h_s_i += len(K1)
#     t_s_i += len(K)
#     if FM.Values["Class_N"] == 'Show':
#         hit_show_v+=1
#     total_show_v+=1
print "Hit_point_V: " + hit_point_v.__str__()
print "Hit_show_V: " + hit_show_v.__str__()
print "Total_point_V: " + total_point_v.__str__()
print "Total_show_V: " + total_show_v.__str__()
print h_p_i
print t_p_i
print h_s_i
print t_s_i
# if FM.Candidate_patch is None:
#     print user + " " + action + " Has no Candidate"
#     continue
# for candidate in FM.Candidate_patch:
#     print candidate.Label
#     cv2.imshow("Objects",candidate.patch)
#     k = cv2.waitKey()
#     k = chr(k & 255)
#     if k == 'y':
#         candidate.Ground_Truth = candidate.Label
#     elif k == 'n':
#         label = raw_input("Label: ")
#         candidate.Ground_Truth = label
#     else:
#         candidate.Ground_Truth = None
# print FM.Candidate_patch[0].Ground_Truth
# # files = gzip.open(path + "/" + user + "_" + action + ".gpz", 'w')
# # cPickle.dump(FM, files, -1)
# # files.close()
