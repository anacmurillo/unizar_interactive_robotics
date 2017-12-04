import math
import multiprocessing
import os
import timeit

import cv2
import joblib
import numpy as np

from Descriptors import *
from Incremental_learning import *
from Interaction import *
from Obj_segment import *

distance = 0.1
NN_max = 150
add = 50
BoW = BoW.BoW("")
Total_set = []
path = "/media/iglu/Data/DatasetIglu"
u = ['user1', 'user2', 'user3', 'user4', 'user5', 'user6', 'user7', 'user8', 'user9', 'user10']  #
a = ['point_1', 'point_2', 'point_3', 'point_4', 'point_5', 'point_6', 'point_7', 'point_8', 'point_9', 'point_10',
     'show_1', 'show_2', 'show_3', 'show_4', 'show_5', 'show_6', 'show_7', 'show_8', 'show_9', 'show_10']
f1 = open("HMM_output.txt", 'w')
Matrix_F = [[0, 0], [0, 0]]
Matrix_V = [[0, 0], [0, 0]]
P_seg = Point_Seg.Point_Seg()
S_seg = Show_Seg.Show_Seg()
C = {'point': 0, 'show': 1}

def func_old(arg):
    nu, na = arg
    user = u[nu]
    action = a[na]
    print user
    print action
    start_time = timeit.default_timer()
    # f1.write("Video: "+user+"_"+action+"\n")
    M = Masking.Masking()
    I_front = Interaction_Recogn.Interaction_Recogn(16, False, 0.6)
    # hmm = HMM.HMM.HMM()
    Np = 0
    Ns = 0
    Votes = []
    Total = []
    Total_set = []
    n_windows = []
    f = open(path + "/" + user + "/" + action + "/speech.txt", 'r')
    s = f.readline()
    s = s.split(" ")
    i = 0
    objectt = s[1]
    if s.__len__() > 2:
        objectt = objectt + s[2]
    if objectt.endswith("\n"):
        objectt = objectt[:-1]

    if not os.path.exists("One_Camera_Front_Fix/" + objectt):
        os.makedirs("One_Camera_Front_Fix/" + objectt)
        os.makedirs("One_Camera_Front_Fix/" + objectt + "/point")
        os.makedirs("One_Camera_Front_Fix/" + objectt + "/show")
    f = open(path + "/" + user + "/" + action + "/k1" + "/List.txt", 'r')
    for line in f:
        Time = line
        file1 = next(f).rstrip('\n')
        file2 = next(f).rstrip('\n')
        Label = next(f).rstrip('\n')
        RGB = cv2.imread(path + "/" + user + "/" + action + "/k1" + "/RGB/" + file1)
        Depth = np.load(path + "/" + user + "/" + action + "/k1" + "/Depth/" + file2)
        Mask = cv2.imread(path + "/" + user + "/" + action + "/k1" + "/MTA/" + file1)
        dep = Depth.copy()

        # Masking Part
        mask, dep, Maski = M.Mask(Mask, dep, RGB)
        # Action Recognition
        RGB_vfil, Rad_angle, Center, Class,P_front = I_front.search_one_image(RGB, Depth, mask, dep, True)
        if Class != None:
            # cv2.imwrite(
            #     "Test/Hand_" + action + "_" + user + "_" + i.__str__() +"_"+Class+ ".jpg",
            #     RGB[Center[0] - 50:Center[0] + 50, Center[1] - 50 : Center[1] + 50, :])
            if Class == 'show':
                Ns += 1
            else:
                Np += 1
            # Votes.append((Class,Label,i))
            # Total.append((RGB, Mask, Depth, dep, RGB_vfil, Rad_angle, Center, i, Maski, user, Label))

            # Class_HMM,Prob = hmm.new_stage(Class,P)
        i += 1
    # if Ns > Np:
    #     for i in Total:
    #         RGB, Mask, Depth, dep, RGB_vfil, Rad_angle, Center, num, Maski, user, Label = i
    #         theta = (Rad_angle / math.pi) * 180.0
    #         O = S_seg.obtain_patch(RGB, Depth, dep, Center, theta,True)
    #         if O.shape[0] != 0 and O.shape[1]!=0:
    #             Total_set.append((O, objectt, user, action))
                # cv2.imwrite(
                #     "One_Camera_Front_Fix/" + objectt + "/show/" + user + "_" + action + "_Frame_" + num.__str__() + ".jpg", O)

    # else:
    #     for i in Total:
    #         RGB, Mask, Depth, dep, RGB_vfil, Rad_angle, Center, num, Maski, user, Label = i
    #         theta = (Rad_angle / math.pi) * 180.0
    #         O = P_seg.draw_triangle(RGB,Depth, Maski, Center, Rad_angle,True)
    #         if O is not None and O.shape[0] != 0 and O.shape[1]!=0:
    #             Total_set.append((O, objectt, user, action))
                # cv2.imwrite(
                #     "One_Camera_Front_Fix/" + objectt + "/point/" + user + "_" + action + "_Frame_" + num.__str__() + ".jpg",
                #     O)
    elapsed = timeit.default_timer() - start_time
    print "Tiempo funcion individual: " + elapsed.__str__()
    return (user,action,i,Ns+Np)

def func(arg):
    I_top = Interaction_Recogn.Interaction_Recogn(16, False, 0.90)
    I_front = Interaction_Recogn.Interaction_Recogn(16, False, 0.90)
    nu, na = arg
    user = u[nu]
    action = a[na]
    print user
    print action
    # Video = Video_saver.Video_saver()
    start_time = timeit.default_timer()
    # f1.write("Video: "+user+"_"+action+"\n")
    M_front = Masking.Masking()
    M_Top = Masking.Masking()
    # hmm = HMM.HMM.HMM()
    Np = 0
    Ns = 0
    Votes = []
    Total = []
    Total_set = []
    f = open(path + "/" + user + "/" + action + "/speech.txt", 'r')
    s = f.readline()
    s = s.split(" ")
    i = 0
    objectt = s[1]
    if s.__len__() > 2:
        objectt = objectt + s[2]
    if objectt.endswith("\n"):
        objectt = objectt[:-1]

    if not os.path.exists("Saved_images_Cand_fix/" + objectt):
        os.makedirs("Saved_images_Cand_fix/" + objectt)
        os.makedirs("Saved_images_Cand_fix/" + objectt + "/point")
        os.makedirs("Saved_images_Cand_fix/" + objectt + "/show")
    f = open(path + "/" + user + "/" + action + "/k2" + "/List.txt", 'r')
    f2 = open(path + "/" + user + "/" + action + "/k1" + "/List.txt", 'r')
    Images = []
    for line in f:
        Time = line
        file1 = next(f).rstrip('\n')
        file2 = next(f).rstrip('\n')
        Label_top = next(f).rstrip('\n')
        RGB_top = cv2.imread(path + "/" + user + "/" + action + "/k2" + "/RGB/" + file1)
        Depth_top = np.load(path + "/" + user + "/" + action + "/k2" + "/Depth/" + file2)
        Mask_top = cv2.imread(path + "/" + user + "/" + action + "/k2" + "/MTA/" + file1)
        dep_top = Depth_top.copy()
        try:
            Time = next(f2).rstrip('\n')
        except StopIteration:
            break
        file3 = next(f2).rstrip('\n')
        file4 = next(f2).rstrip('\n')
        Label_front = next(f2).rstrip('\n')
        RGB_front = cv2.imread(path + "/" + user + "/" + action + "/k1" + "/RGB/" + file3)
        Depth_front = np.load(path + "/" + user + "/" + action + "/k1" + "/Depth/" + file4)
        Mask_front = cv2.imread(path + "/" + user + "/" + action + "/k1" + "/MTA/" + file3)
        dep_front = Depth_front.copy()

        # Masking Part
        mask_top, dep_top, Maski_top = M_Top.Mask(Mask_top, dep_top, RGB_top)
        mask_front, dep_front, Maski_front = M_front.Mask(Mask_front, dep_front, RGB_front)
        SD=False
        if i==35:
            SD = True
        # Action Recognition
        RGB_vfil_front, Rad_angle_front, Center_front, Class_front, P_front = I_front.search_one_image(RGB_front, Depth_front, mask_front, dep_front, SD)
        RGB_vfil_top, Rad_angle_top, Center_top, Class_top, P_top = I_top.search_one_image(RGB_top, Depth_top, mask_top, dep_top, False)
        Images.append((RGB_top,Mask_top,RGB_front,Mask_front))
        if Class_top != None or Class_front != None:
            if Class_front == None:
                if Class_top == 'show':
                    Ns += 1
                else:
                    Np += 1
                Votes.append((Class_top, Label_top, i))
                Matrix_F[C[Label_top]][C[Class_top]] += 1
                Total.append((RGB_top, Mask_top, Depth_top, dep_top, Rad_angle_top, Center_top, i, Maski_top, user, Label_top,RGB_front,Mask_front,None,None,None, True))
            elif Class_top == None:
                if Class_front == 'show':
                    Ns += 1
                else:
                    Np += 1
                Votes.append((Class_front, Label_front, i))
                Matrix_F[C[Label_front]][C[Class_front]] += 1
                Total.append((RGB_front, Mask_front, Depth_front, dep_front, Rad_angle_front, Center_front, i, Maski_front, user, Label_front,RGB_top,Mask_top,None,None,None,False))
            else:
                if Class_top == 'show':
                    if Class_top == Class_front:
                        Ns+=1
                        Votes.append((Class_front, Label_front, i))
                        Matrix_F[C[Label_front]][C[Class_front]] += 1
                        Total.append((
                                     RGB_front, Mask_front, Depth_front, dep_front, Rad_angle_front,
                                     Center_front, i,Maski_front, user, Label_front, RGB_top, Mask_top,Center_top,Depth_top, dep_top, False))
                    elif P_top >= P_front:
                        Ns += 1
                        Votes.append((Class_top, Label_top, i))
                        Matrix_F[C[Label_top]][C[Class_top]] += 1
                        Total.append((
                            RGB_top, Mask_top, Depth_top, dep_top, Rad_angle_top, Center_top,i,
                            Maski_top, user, Label_top, RGB_front, Mask_front,Center_front,Depth_front,dep_front, True))
                    else:
                        Np += 1
                        Votes.append((Class_front, Label_front, i))
                        Matrix_F[C[Label_front]][C[Class_front]] += 1
                        Total.append((
                            RGB_front, Mask_front, Depth_front, dep_front,Rad_angle_front,
                            Center_front,i,  Maski_front, user, Label_front, RGB_top, Mask_top,None,None,None, False))
                else:
                    if Class_top == Class_front:
                        Np+=1
                        Votes.append((Class_front, Label_front, i))
                        Matrix_F[C[Label_front]][C[Class_front]] += 1
                        Total.append((
                                     RGB_front, Mask_front, Depth_front, dep_front, Rad_angle_front,
                                     Center_front,i,  Maski_front, user, Label_front, RGB_top, Mask_top,None,None,None, False))
                    else:
                        Ns += 1
                        Votes.append((Class_front, Label_front, i))
                        Matrix_F[C[Label_front]][C[Class_front]] += 1
                        Total.append((
                            RGB_front, Mask_front, Depth_front, dep_front, Rad_angle_front,
                            Center_front,i, Maski_front, user, Label_front, RGB_top, Mask_top,Center_top,Depth_top, dep_top, False))
            # Class_HMM,Prob = hmm.new_stage(Class,P)
        i += 1
    Images = Images[:5]
    if Ns > Np:
        Num = 0
        for i in Total:
            RGB, Mask, Depth, dep, Rad_angle, Center,num, Maski, user, Label,RGB2,Mask2,Center2,Depth2, dep2,isTop = i
            theta = (Rad_angle / math.pi) * 180.0
            if isTop:
                O2 = S_seg.obtain_patch(RGB, Depth, dep, Center, theta, True)
                if(Depth2 is not None):
                    O = S_seg.obtain_patch(RGB2, Depth2, dep2, Center2, 0.0, True)
                else:
                    O = None
            else:
                O = S_seg.obtain_patch(RGB, Depth, dep, Center, theta, True)
                if(Depth2 is not None):
                    O2 = S_seg.obtain_patch(RGB2, Depth2, dep2, Center2, 0.0, True)
                else:
                    O2 = None
            Total_set.append((O, O2, objectt, user, action))
            if O2 is not None:
                cv2.imwrite(
                    "Saved_images_Cand_fix/" + objectt + "/show/" + user + "_" + action + "_Top_Frame_" + num.__str__() + ".jpg",
                    O2)
            if O is not None:
                cv2.imwrite(
                    "Saved_images_Cand_fix/" + objectt + "/show/" + user + "_" + action + "_Front_Frame_" + num.__str__() + ".jpg",
                    O)

    else:
        Candidates = Obj_Cand.get_candidate(Images)
        # S_seg.add_candidates(Candidates)
        P_seg.add_candidates(Candidates)
        print "Candidatos " + user + " " + action + ":" + Candidates.__len__().__str__()
        if Candidates.__len__() == 0:
            return Total_set
        Lista = [0] * Candidates.__len__()
        for i in Total:
            RGB, Mask, Depth, dep, Rad_angle, Center, num, Maski, user, Label,RGB2,Mask2,Center_top,D,D,isTop = i
            theta = (Rad_angle / math.pi) * 180.0
            Number = P_seg.obtain_candidate(Center, Rad_angle,isTop)
            if Number is not None:
                Lista[Number] += 1
        M = Lista.index(max(Lista))
        Can = Candidates[M][0]
        Can.Fixed()
        Can_top = Candidates[M][1]
        Can_top.Fixed()
        Num = 0
        for i in Total:
            RGB, Mask, Depth, dep, Rad_angle, Center, num, Maski, user, Label, RGB2, Mask2,Center_top,D,D, isTop = i
            if isTop:
                RGB = Obj_Cand.rotateImage(RGB, Obj_Cand.angulo, Obj_Cand.centro)
                O = RGB2[Can.bottom:Can.top, Can.left:Can.right]
                O2 = RGB[Can_top.bottom:Can_top.top, Can_top.left:Can_top.right]
                Total_set.append((O, O2, objectt, user, action))
            else:
                RGB2 = Obj_Cand.rotateImage(RGB2, Obj_Cand.angulo, Obj_Cand.centro)
                O = RGB[Can.bottom:Can.top, Can.left:Can.right]
                O2 = RGB2[Can_top.bottom:Can_top.top, Can_top.left:Can_top.right]
                Total_set.append((O, O2, objectt, user, action))
            cv2.imwrite(
                "Saved_images_Cand_fix/" + objectt + "/point/" + user + "_" + action + "_Top_Frame_" + num.__str__() + ".jpg",
                O2)
            cv2.imwrite(
                "Saved_images_Cand_fix/" + objectt + "/point/" + user + "_" + action + "_Front_Frame_" + num.__str__() + ".jpg",
                O)

    elapsed = timeit.default_timer() - start_time
    print "Tiempo funcion individual: " + elapsed.__str__()
    return Votes,Matrix_F,Matrix_V

def In_Patch():
    print "EMPIEZA"
    start_time1 = timeit.default_timer()
    # R = [func((aa, bb)) for aa in range(10) for bb in range(20)]
    z = [(aa, bb) for aa in range(10) for bb in range(20)]
    pool = multiprocessing.Pool(6)
    R = pool.map(func_old, z,1)
    pool.close()
    pool.join()
    elapsed = timeit.default_timer() - start_time1
    print "Tiempo: " + elapsed.__str__()

    # for nu in xrange(10):
    #     for na in xrange(20):
    #         user = u[nu]
    #         action = a[na]
    #         print user
    #         print action
    #         start_time = timeit.default_timer()
    #         # f1.write("Video: "+user+"_"+action+"\n")
    #         M = Masking.Masking()
    #         # hmm = HMM.HMM.HMM()
    #         Np = 0
    #         Ns = 0
    #         Total = []
    #         f= open(path + "/" + user + "/" + action + "/speech.txt", 'r')
    #         s = f.readline()
    #         s = s.split(" ")
    #         i=0
    #         objectt = s[1]
    #         if s.__len__() >2:
    #             objectt = objectt+s[2]
    #         if objectt.endswith("\n"):
    #             objectt = objectt[:-1]
    #
    #         if not os.path.exists("Saved_images/"+objectt):
    #             os.makedirs("Saved_images/"+objectt)
    #             os.makedirs("Saved_images/" + objectt+"/point")
    #             os.makedirs("Saved_images/" + objectt+"/show")
    #         f = open(path + "/" + user + "/" + action + "/k1" + "/List.txt", 'r')
    #         for line in f:
    #             Time = line
    #             file1 = next(f).rstrip('\n')
    #             file2 = next(f).rstrip('\n')
    #             Label = next(f).rstrip('\n')
    #             RGB = cv2.imread(path + "/" + user + "/" + action + "/k1" + "/RGB/" + file1)
    #             Depth = np.load(path + "/" + user + "/" + action + "/k1" + "/Depth/" + file2)
    #             Mask = cv2.imread(path + "/" + user + "/" + action + "/k1" + "/MTA/" + file1)
    #             dep = Depth.copy()
    #
    #             # Masking Part
    #             mask,dep,Maski = M.Mask(Mask,dep,RGB)
    #
    #             # Action Recognition
    #             RGB_vfil,Rad_angle,Center,Class,P =I.search_one_image(RGB, Depth,mask,dep, True)
    #             if Class!=None:
    #                 if Class == 'show':
    #                     Ns+=1
    #                 else:
    #                     Np+=1
    #                 Matrix_F[C[Label]][C[Class]]+=1
    #                 Total.append((RGB,Mask,Depth,dep,RGB_vfil,Rad_angle,Center,i,Maski,user,Label))
    #
    #                 # Class_HMM,Prob = hmm.new_stage(Class,P)
    #             i+=1
    #         if Ns > Np:
    #             for i in Total:
    #                 RGB, Mask, Depth, dep, RGB_vfil,Rad_angle,Center,num ,Maski,user,Label= i
    #                 theta = (Rad_angle / math.pi) * 180.0
    #                 O = S_seg.obtain_patch(RGB,Depth,dep,Center,theta)
    #                 Total_set.append((O,objectt,user,Label))
    #                 cv2.imwrite("Saved_images/" + objectt + "/show/" + user + "_" + action + "_Frame_" + num.__str__() + ".jpg", O)
    #
    #         else:
    #             for i in Total:
    #                 RGB, Mask, Depth, dep, RGB_vfil, Rad_angle, Center,num ,Maski,user,Label= i
    #                 theta = (Rad_angle / math.pi) * 180.0
    #                 O = P_seg.draw_triangle(RGB, Maski, Center, Rad_angle)
    #                 if O != None:
    #                     Total_set.append((O, objectt,user,Label))
    #                     cv2.imwrite("Saved_images/" + objectt + "/point/" + user + "_" + action + "_Frame_" + num.__str__() + ".jpg", O)
    #
    #         elapsed = timeit.default_timer() - start_time
    #         print "Tiempo: " + elapsed.__str__()
    # Total = [item for sublist in R for item in sublist]
    joblib.dump((R,Matrix_F,Matrix_V),"Data_Front_camera_v5.pkl",compress=9)
    return R


def spliting(Test_n, Porcen):
    X_train = None
    Y_train = None
    M_train = None
    Z_train = None
    N = 0
    for i in xrange(Porcen):
        if N == Test_n:
            N += 1
        T = joblib.load('/home/iglu/catkin_ws/src/IL_Complete/src/Test_' + N.__str__() + '.pkl')
        np.random.shuffle(T)
        X = T[:, 0]
        Y = T[:, 1]
        Z = T[:, 2]
        M = T[:, 3]
        if X_train is None:
            X_train = X
            Y_train = Y
            Z_train = Z
            M_train = M
        else:
            X_train = np.hstack((X_train, X))
            Y_train = np.hstack((Y_train, Y))
            Z_train = np.hstack((Z_train, Z))
            M_train = np.hstack((M_train, M))
    T = joblib.load('/home/iglu/catkin_ws/src/IL_Complete/src/Test_' + Test_n.__str__() + '.pkl')
    X = T[:, 0]
    Y = T[:, 1]
    X_test = X.tolist()
    Y_test = Y.tolist()
    return X_train.tolist(), X_test, Y_train.tolist(), Y_test,Z_train,M_train

def NN_S(arg):#NN_max,last_test,Perc):
    print arg
    NN_max,last_test,Perc= arg
    start_time = timeit.default_timer()
    # print "NN_MAX = "+NN_max.__str__()
    # print "Percentage: "+Perc.__str__()+'0'
    # print "Test block : "+last_test.__str__()
    BoW.loadBoW("Descriptors/BoW_ORB.pkl")
    D = Descriptors.Descriptors()
    M_HC = ConfusionMatrix.ConfusionMatrix()
    M_BoW = ConfusionMatrix.ConfusionMatrix()
    Nearest_HC = NN.NN(distance, 8 * 3, NN_max)
    Nearest_BoW = NN.NN(distance, 1000, NN_max)
    print "Step 1: "+NN_max.__str__()+" "+Perc.__str__()+'0 '+last_test.__str__()
    # X_train, X_test, y_train, y_test =train_test_split(Total_set[:,0],Total_set[:,1],test_size=0.10, random_state=42)
    X_train, X_test, y_train, y_test,Z_train,M_train = spliting(last_test,Perc)
    for n in xrange(len(X_train)):
        final = X_train[n]
        if final.shape[1] == 0 or final.shape[0] == 0:
            continue
        label = y_train[n]
        # print label
        # print M_train[n]
        # print Z_train[n]
        X, img2 = D.ComputeORB(final, None)
        if X is not None:
            Obj = []
            Obj.append((" ", X))
            Hist_BoW = BoW.testwoBoW(None, Obj, 1)
            Hist_BoW = Hist_BoW[0][:]
            out2 = Nearest_BoW.train(None, Hist_BoW, label, 1)
        else:
            # print "ERROR NO BRIEF POINTs"
            Hist_BoW = []
        Hist_HC = D.ComputeHC(final, None)
        Hist_HC = Hist_HC[0][:]
        out1 = Nearest_HC.train(None, Hist_HC, label, 1)
    # print "INFO NEAREST HC"
    # Nearest_HC.info()
    # print "INFO NEAREST BOW"
    # Nearest_BoW.info()
    print "Step 2: " + NN_max.__str__() + " " + Perc.__str__() + '0 ' + last_test.__str__()
    for n in xrange(len(X_test)):
        final = X_test[n]
        label = y_test[n]
        if final.shape[1] == 0 or final.shape[0] == 0:
            continue
        X, img2 = D.ComputeORB(final, None)
        if X is not None:
            Obj = []
            Obj.append((" ", X))
            Hist_BoW = BoW.testwoBoW(None, Obj, 1)
            Hist_BoW = Hist_BoW[0][:]
            outBoW,Label_Bow,d_BoW = Nearest_BoW.test(None,Hist_BoW,0)
            M_BoW.add_pair(label,Label_Bow)
        else:
            # print "ERROR NO BRIEF POINTs"
            Hist_BoW = []
        Hist_HC = D.ComputeHC(final, None)
        Hist_HC = Hist_HC[0][:]
        outHC, Label_HC, d_HC = Nearest_HC.test(None, Hist_HC, 0)
        M_HC.add_pair(label,Label_HC)
    print "Step 3: " + NN_max.__str__() + " " + Perc.__str__() + '0 ' + last_test.__str__()
    # print "Matrix BoW"
    # M_BoW.show_info("Matrix_BoW.jpg",False)
    # print "Matrix HC"
    Out_NN = Nearest_HC.info(False)
    Out_CM = M_HC.show_info("Matrix_HC.jpg",False)
    elapsed = timeit.default_timer() - start_time
    print "Tiempo: " + elapsed.__str__()
    T = []
    T.append((NN_max,Perc*10,last_test,Out_NN,Out_CM))
    return T


# import cProfile
# pr = cProfile.Profile()
# pr.enable()
start_time1 = timeit.default_timer()
func((4,8))
# In_Patch()
# pr.disable()
# pr.print_stats(sort="time")
# Total_set = joblib.load("/home/iglu/catkin_ws/src/IL_Complete/src/Patches_obtainv3.pkl")
# Total_set = np.array(Total_set)
# Names = np.unique(Total_set[:, 1])
# for name in Names:
#     Cl = Total_set[Total_set[:, 1] == name]
#     np.random.shuffle(Cl)
#     joblib.dump(Cl, '/home/iglu/catkin_ws/src/IL_Complete/src/Separate/Class_' + name + '.pkl', compress=9)
# -------------------------------------------------------------------------------------
# for i in os.listdir('/home/iglu/catkin_ws/src/IL_Complete/src/Separate'):
#     T = joblib.load('/home/iglu/catkin_ws/src/IL_Complete/src/Separate/' + i)
#     n = T.shape[0] % 10
#     if n != 0:
#         T = T[:-n]
#     joblib.dump(T, '/home/iglu/catkin_ws/src/IL_Complete/src/Separate/' + i, compress=9)
# -------------------------------------------------------------------------------------
# Division = [None, None, None, None, None, None, None, None, None, None]
# for i in os.listdir('/home/iglu/catkin_ws/src/IL_Complete/src/Separate'):
#     T = joblib.load('/home/iglu/catkin_ws/src/IL_Complete/src/Separate/' + i)
#     if Division[0] == None:
#         for n in xrange(10):
#             Division[n] = np.split(T, 10)[n]
#     else:
#         for n in xrange(10):
#             D = Division[n]
#             Division[n] = np.vstack((D, np.split(T, 10)[n]))
# for n in xrange(10):
#     joblib.dump(Division[n], '/home/iglu/catkin_ws/src/IL_Complete/src/Separate/Test_' + n.__str__() + '.pkl',
#                 compress=9)

# z = [NN_S(aa, bb,cc) for aa in range(50,500,50) for bb in range(0,10,1) for cc in range(1,10,1)]
# joblib.dump(z,"/home/iglu/catkin_ws/src/IL_Complete/src/Output_Experiments_vlineal.pkl",compress=9)
elapsed = timeit.default_timer() - start_time1
print "Tiempo Total: " + elapsed.__str__()
exit(0)
# pr = cProfile.Profile()
# pr.enable()
#######################################################################
# start_time1 = timeit.default_timer()
# z = [(aa, bb,cc) for aa in range(300,500,50) for bb in range(0,10,1) for cc in range(1,10,1)]
# print z
# pool = multiprocessing.Pool(10)
# R = pool.map(NN_S, z)
# print "Final:"
# print R
# pool.close()
# pool.join()
# elapsed = timeit.default_timer() - start_time1
# print "Tiempo Total: " + elapsed.__str__()
# joblib.dump(R,"Output_Experiments.pkl",compress=9)
#######################################################################
# for mm in xrange(5):
#     NN_max += add
#     for o in xrange(10):
#         last_test=9-o
#         perc = 1
#         add_perc = 1
#         for i in xrange(8):
#             perc = perc+add_perc
#             NN_S(NN_max,perc)
# pr.disable()
# pr.print_stats(sort="time")
    # if Class == 'Point':64.33
    # Po+=1
    #     if center != None:
    #         RGB2 = P.draw_triangle(RGB,Mask,center,theta)
    #         RGB[200:480,0:640]=RGB2
    #     R = np.hstack((R,RGB))
    #     cv2.putText(R,user+"_"+action, (20,20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #     if theta != None:
    #         theta = (theta / math.pi) * 180.0
    #     else:
    #         theta = "No angle"
    #     cv2.putText(R, theta.__str__(), (640, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # elif Class == 'Show':
    #     Sh+=1
    #
    #     R = np.hstack((R,RGB))

