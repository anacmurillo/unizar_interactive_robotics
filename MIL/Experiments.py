import cv2
import cv
import sys
import os
import timeit
from Offline_Learning import SVM
from Offline_Learning import NN_offline
import sys
import math
import HMM.HMM
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from Descriptors import *
from Incremental_learning import *
import numpy as np
import joblib
import multiprocessing
M_HC = ConfusionMatrix.ConfusionMatrix()
M_BoW = ConfusionMatrix.ConfusionMatrix()
M_DL = ConfusionMatrix.ConfusionMatrix()
BoW = BoW.BoW("")
BoW.loadBoW("Descriptors/BoW_ORB.pkl")
D = Descriptors.Descriptors()

def Train(Train_set,Name_set,Method):
    if Method == 'SVM':
        Classificator_Bow = SVM.SVM_offline("Offline_Learning/SVM_BoW.pkl",100,0.001, 50000000, False, False)
        Classificator_HC = SVM.SVM_offline("Offline_Learning/SVM_HC.pkl",100,0.001, 50000000, False, False)
    else:
        Classificator_Bow = NN_offline.NN_offline("Offline_Learning/NN_BoW.pkl",20)
        Classificator_HC = NN_offline.NN_offline("Offline_Learning/NN_HC.pkl", 20)
    # print "Step 1: "
    Train_Hc = np.array([])
    Train_Bow= np.array([])
    Label_Hc = np.array([])
    Label_Bow= np.array([])
    # X_train, X_test, y_train, y_test =train_test_split(Total_set[:,0],Total_set[:,1],test_size=0.10, random_state=42)
    for n in xrange(len(Train_set)):
        final = Train_set[n]
        if final.shape[1] == 0 or final.shape[0] == 0:
            continue
        label = Name_set[n]
        # print label
        # print M_train[n]
        # print Z_train[n]
        X, img2 = D.ComputeORB(final, None)
        if X is not None:
            Obj = []
            Obj.append((" ", X))
            Hist_BoW = BoW.testwoBoW(None, Obj, 1)
            Hist_BoW = Hist_BoW[0][:]
            if Train_Bow.shape[0] == 0:
                Train_Bow = Hist_BoW
                Label_Bow = np.array([label])
            else:
                Train_Bow = np.vstack((Train_Bow,Hist_BoW))
                Label_Bow = np.vstack((Label_Bow,np.array([label])))
        else:
            # print "ERROR NO BRIEF POINTs"
            Hist_BoW = []
            out2 = ""
        Hist_HC = D.ComputeHC(final, None)
        Hist_HC = Hist_HC[0][:]
        if Train_Hc.shape[0] == 0:
            Train_Hc= Hist_HC
            Label_Hc = np.array([label])
        else:
            Train_Hc = np.vstack((Train_Hc, Hist_HC))
            Label_Hc = np.vstack((Label_Hc, np.array([label])))

    out2 = Classificator_Bow.train(Train_Bow, Label_Bow)
    out1 = Classificator_HC.train(Train_Hc, Label_Hc)
    return Classificator_Bow,Classificator_HC


def Test(Test_set,GT_set,Classificator_Bow,Classificator_HC,Method):
    def Evaluate(final,Classificator_Bow,Classificator_HC,Method):
        if final.shape[1] == 0 or final.shape[0] == 0:
            return None,None
        X, img2 = D.ComputeORB(final, None)
        if X is not None:
            Obj = []
            Obj.append((" ", X))
            Hist_BoW = BoW.testwoBoW(None, Obj, 1)
            Hist_BoW = Hist_BoW[0][:]
            Label = Classificator_Bow.test(Hist_BoW)
            if Method == 'SVM':
                # Label,P_Bow = Label
                P_Bow =1
                Label_Bow = Label[0]
            else:
                P_Bow=1
                Label_Bow = Label[0][0]
        else:
            # print "ERROR NO BRIEF POINTs"
            Label_Bow = None
            P_Bow = 1
        Hist_HC = D.ComputeHC(final, None)
        Hist_HC = Hist_HC[0][:]
        Label = Classificator_HC.test(Hist_HC)
        if Method == 'SVM':
            Label = Label
            P_HC = 1
            Label_HC = Label[0]
        else:
            P_HC=1
            Label_HC = Label[0][0]
        return Label_Bow,P_Bow,Label_HC,P_HC

    # print "INFO NEAREST HC"
    # Nearest_HC.info()
    # print "INFO NEAREST BOW"
    # Nearest_BoW.info()
    # print "Step 2: "
    # M_HC = ConfusionMatrix.ConfusionMatrix()
    # M_BoW = ConfusionMatrix.ConfusionMatrix()
    # print "Step 2: " + NN_max.__str__() + " " + Perc.__str__() + '0 ' + last_test.__str__()
    for n in xrange(len(Test_set)):
        final = Test_set[n]
        label = GT_set[n]
        if not Classificator_Bow.IsObject(label) and not Classificator_HC.IsObject(label):
            # print label+" is not found"
            continue
        if final.__class__ != tuple:
            Label_Bow,P_Bow,Label_HC,P_HC = Evaluate(final,Classificator_Bow,Classificator_HC,Method)
            M_HC.add_pair(label, Label_HC)
            if Label_Bow is not None:
                M_BoW.add_pair(label, Label_Bow)
        else:
            final1 = final[0]
            final2 = final[1]
            Label_Bow_1, P_Bow_1, Label_HC_1, P_HC_1 = Evaluate(final1, Classificator_Bow, Classificator_HC, Method)
            Label_Bow_2, P_Bow_2, Label_HC_2, P_HC_2 = Evaluate(final2, Classificator_Bow, Classificator_HC, Method)
            # print label
            # print Label_Bow_1
            # print P_Bow_1
            # print Label_Bow_2
            # print P_Bow_2
            # print Label_HC_1
            # print P_HC_1
            # print Label_HC_2
            # print P_HC_2
            if Label_Bow_1 is None and Label_Bow_2 is not None:
                M_BoW.add_pair(label, Label_Bow_2)
            elif Label_Bow_1 is not None and Label_Bow_2 is None:
                M_BoW.add_pair(label, Label_Bow_1)
            elif Label_Bow_1 is not None and Label_Bow_2 is not None:
                if Label_Bow_1 == Label_Bow_2:
                    M_BoW.add_pair(label, Label_Bow_1)
                elif P_Bow_1 >= P_Bow_2:
                    M_BoW.add_pair(label, Label_Bow_1)
                else:
                    M_BoW.add_pair(label, Label_Bow_2)
            if Label_HC_1 == Label_HC_2:
                M_HC.add_pair(label, Label_HC_1)
            elif P_HC_1 >= P_HC_2:
                M_HC.add_pair(label, Label_HC_1)
            else:
                M_HC.add_pair(label, Label_HC_2)
    # print "Step 3: " + NN_max.__str__() + " " + Perc.__str__() + '0 ' + last_test.__str__()
    print "Matrix BoW"
    Out_BoW = M_BoW.show_info("Matrix_NN_BoW.jpg",False)
    print "Matrix HC"
    # Out_NN = Nearest_HC.info(False)
    Out_HC = M_HC.show_info("Matrix_NN_HC.jpg",False)
    # elapsed = timeit.default_timer() - start_time
    # print "Tiempo: " + elapsed.__str__()
    T = []
    T.append((Out_BoW,Out_HC))
    return T


def Train_DL(Train_set,Name_set,Method):
    if Method == 'SVM':
        Classificator = SVM.SVM_offline("Offline_Learning/SVM_DL.pkl",100,0.01, 5000000, False, True)
    else:
        Classificator = NN_offline.NN_offline("Offline_Learning/NN_DL.pkl",20)
    # print "Step 1: "
    Train = np.array([])
    Label = np.array([])
    for n in xrange(len(Train_set)):
        final = Train_set[n]
        label = Name_set[n]
        if Train.shape[0] == 0:
            Train= final
            Label = np.array([label])
        else:
            Train = np.vstack((Train, final))
            Label = np.vstack((Label, np.array([label])))
    out = Classificator.train(Train, Label)
    return Classificator


def Test_DL(Test_set,GT_set,Classificator,Method):
    M_DL = ConfusionMatrix.ConfusionMatrix()
    for n in xrange(len(Test_set)):
        final = Test_set[n]
        label = GT_set[n]
        Label = Classificator.test(final)
        if Method == 'SVM':
            Label_DL = Label[0]
        else:
            Label_DL = Label[0][0]
        M_DL.add_pair(label, Label_DL)
    # print "Matrix DL"
    return M_DL.show_info("Matrix_DL_NN.jpg", False)

def Experiment(Train_data,Test_data,Test_user,Method):

    def Obtain_data_DL(Train,test,user,Wash):
        Users = ['user1', 'user2', 'user3', 'user4', 'user5', 'user6', 'user7', 'user8', 'user9', 'user10']
        Training = []
        Training_Name = []
        Testing = []
        Testing_Name = []
        if user is not None:
            user = Users[user]
        i = 0
        if Wash:
            f = open(Train + "/List.txt", 'r')
            Data = cv.Load(Train + "/feats_fc8.xml")
            Data = np.array(Data)
            for line in f:
                user_D = line.split('/')[1]
                if user_D != user or user is None:
                    feature = Data[i]
                    Training.append(feature)
                    Training_Name.append(line.split(" ")[1].rstrip())
                i += 1
            i = 0
            f.close()
            skf = StratifiedShuffleSplit(n_splits=10,test_size = 0.1)
            T2 = []
            t=0
            T2_name = []
            for train,Test in skf.split(Training,Training_Name):
                for n in xrange(len(train)):
                    T2.append(Training[train[n]])
                    T2_name.append(Training_Name[train[n]])
                for n in xrange(len(Test)):
                    Testing.append(Training[Test[n]])
                    Testing_Name.append(Training_Name[Test[n]])
                Classificator = Train_DL(T2, T2_name, Meth)
                    # print "Testing data..."
                Output = Test_DL(Testing, Testing_Name, Classificator, Meth)
                print "Ending Experiment with parameters " + Train_data.__str__() + "_" + Test_data.__str__() + "_" + t.__str__() + "_" + Method.__str__()
                t+=1
            exit(0)
            # return T2, T2_name, Testing, Testing_Name
        f= open(Train+"/List.txt",'r')
        Data = cv.Load(Train+"/feats_fc8.xml")
        Data = np.array(Data)
        for line in f:
            user_D = line.split('/')[1]
            if user_D != user or user is None:
                feature = Data[i]
                Training.append(feature)
                Training_Name.append(line.split(" ")[1].rstrip())
            i+=1
        i=0
        f.close()
        f= open(test+"/List.txt",'r')
        Data = cv.Load(test+"/feats_fc8.xml")
        Data = np.array(Data)
        for line in f:
            user_D = line.split('/')[1]
            if user_D == user or user is None:
                feature = Data[i]
                Testing.append(feature)
                Testing_Name.append(line.split(" ")[1].rstrip())
            i+=1
        print Training.__len__()
        print Testing.__len__()
        return Training,Training_Name,Testing,Testing_Name


    def Obtain_data(Train,Test,User,Wash,s1,s2):
        Users = ['user1','user2','user3','user4','user5','user6','user7','user8','user9','user10']
        Training = []
        Training_Name = []
        Testing =[]
        Testing_Name =[]
        if Wash:
            i = 0
            f = open(Train + "/List.txt", 'r')
            for line in f:
                feature = cv2.imread(Train+"/"+line[2:].split(" ")[0])
                if feature is None:
                    print Test+"/"+line[2:].split(" ")[0]
                    continue
                Training.append(feature)
                Training_Name.append(line.split(" ")[1].rstrip())
                i += 1
            n = 0
            for i in xrange(10):
                for x in os.listdir(Test+"/"+Users[i]):
                    Splited = x[:-4].split('_')
                    Label = Splited[-1]
                    RGB = cv2.imread(Test+"/"+Users[i]+"/"+x)
                    if RGB is None:# and x.__contains__("Top"):
                        continue
                    n+=1
                    Testing.append(RGB)
                    Testing_Name.append(Label)
        elif User is None:
            for i in xrange(10):
                for x in os.listdir(Train+"/"+Users[i]):
                    Splited = x[:-4].split('_')
                    Label = Splited[-1]
                    RGB = cv2.imread(Train+"/"+Users[i]+"/"+x)
                    if RGB is None and x.__contains__("Top"):
                        print x
                        continue
                    Training.append(RGB)
                    Training_Name.append(Label)
                for x in os.listdir(Test+"/"+Users[i]):
                    Splited = x[:-4].split('_')
                    Label = Splited[-1]
                    RGB = cv2.imread(Test+"/"+Users[i]+"/"+x)
                    if RGB is None and x.__contains__("Top"):
                        continue
                    Testing.append(RGB)
                    Testing_Name.append(Label)
        else:
            for i in xrange(10):
                if i != User:
                    for x in os.listdir(Train + "/" + Users[i]):
                        Splited = x[:-4].split('_')
                        Label = Splited[-1]
                        # if x.__contains__("Front"):
                        #     x2 = x.replace("Front","Top")
                        RGB = cv2.imread(Train + "/" + Users[i] + "/" + x)
                        #     RGB2 = cv2.imread(Train + "/" + Users[i] + "/" + x2)
                        #     if RGB2 is None:
                        #         RGB = RGB1
                        #     else:
                        #         RGB = (RGB1,RGB2)
                        # if x.__contains__("Top"):
                        #     continue
                        Training.append(RGB)
                        Training_Name.append(Label)
                else:
                    for x in os.listdir(Test + "/" + Users[i]):
                        Splited = x[:-4].split('_')
                        Label = Splited[-1]
                        if x.__contains__("Front") and x.__contains__("point"):
                            x2 = x
                            x2 = x2.replace("Front","Top")
                            RGB1 = cv2.imread(Test + "/" + Users[i] + "/" + x)
                            RGB2 = cv2.imread(Test + "/" + Users[i] + "/" + x2)
                            if RGB2 is None:
                                RGB = cv2.imread(Test + "/" + Users[i] + "/" + x)
                            else:
                                RGB = (RGB1,RGB2)
                        elif x.__contains__("Top") or x.__contains__("show"):
                            continue
                        else:
                            RGB = cv2.imread(Test + "/" + Users[i] + "/" + x)
                        Testing.append(RGB)
                        Testing_Name.append(Label)
        print Training.__len__()
        print Testing.__len__()
        return Training,Training_Name,Testing,Testing_Name

    Data = {-1:'/media/iglu/Data/Trainting_test_set/Wash2',
            0:'/media/iglu/Data/Trainting_test_set/Manual',
            1:'/media/iglu/Data/Trainting_test_set/One_Cam_Front',
            2:'/media/iglu/Data/Trainting_test_set/One_Cam_Top',
            3:'/media/iglu/Data/Trainting_test_set/Two_Cam_3',
            4: '/media/iglu/Data/Trainting_test_set/Wash2_Fixed',
            5: '/media/iglu/Data/Trainting_test_set/Manual_Fixed_size',
            6: '/media/iglu/Data/Trainting_test_set/One_Cam_Front_Fixed_Size',
            7: '/media/iglu/Data/Trainting_test_set/Two_Cam_Fixed'}

    if Method == 1:
        Meth = 'SVM'
    else:
        Meth = 'NN'
    if Train_data == -1:
        Train_path = Data[Train_data]
        Test_path = Data[Test_data]
        # print "Obtaining data..."
        Train_set, Train_Name, Test_set, Test_name = Obtain_data(Train_path, Test_path, Test_user, True,'_','_')
        # print "Training data..."
        Classificator_Bow, Classificator_HC = Train(Train_set, Train_Name, Meth)
        # print "Testing data..."
        Output = Test(Test_set, Test_name, Classificator_Bow, Classificator_HC, Meth)
        print "Ending Experiment with parameters " + Train_data.__str__() + "_" + Test_data.__str__() + "_" + Test_user.__str__() + "_" + Method.__str__()
        return Output
    if Train_data <=3 or Test_data <= 3:
        Train_path = Data[Train_data]
        Test_path = Data[Test_data]
        # print "Obtaining data..."
        # Train_set,Train_Name,Test_set_p,Test_name_p = Obtain_data(Train_path,Test_path,Test_user,False,'_','show')
        Train_set, Train_Name, Test_set_s, Test_name_s = Obtain_data(Train_path, Test_path, Test_user, False, '_','_')
        # print "Training data..."
        if Test_set_s == []:
            return None
        Classificator_Bow, Classificator_HC = Train(Train_set,Train_Name,Meth)
        # print "Testing data..."
        # print "Point"
        #     Output = Test(Test_set_p,Test_name_p,Classificator_Bow,Classificator_HC,Meth)
        # print "Show"
        Output = Test(Test_set_s, Test_name_s, Classificator_Bow, Classificator_HC, Meth)
        print "Ending Experiment with parameters "+Train_data.__str__()+"_"+Test_data.__str__()+"_"+Test_user.__str__()+"_"+Method.__str__()
        return Output
    else:
        Train_path = Data[Train_data]
        Test_path = Data[Test_data]
        # print "Obtaining data..."
        Train_set, Train_Name, Test_set, Test_name = Obtain_data_DL(Train_path, Test_path, Test_user,True)
        # print "Training data..."
        Classificator = Train_DL(Train_set, Train_Name, Meth)
        # print "Testing data..."
        Output = Test_DL(Test_set, Test_name, Classificator, Meth)
        print "Ending Experiment with parameters " + Train_data.__str__() + "_" + Test_data.__str__() + "_" + Test_user.__str__() + "_" + Method.__str__()
        return Output
    # else:
    #     Train_path = Data[Train_data]
    #     Test_path = Data[Test_data]
    #     # print "Obtaining data..."
    #     Train_set, Train_Name, Test_set, Test_name = Obtain_data(Train_path, Test_path, None)
    #     # print "Training data..."
    #     Classificator_Bow, Classificator_HC = Train(Train_set, Train_Name, Meth)
    #     # print "Testing data..."
    #     Output = Test(Test_set, Test_name, Classificator_Bow, Classificator_HC, Meth)
    #     print "Ending Experiment with parameters " + Train_data.__str__() + "_" + Test_data.__str__() + "_" + Test_user.__str__() + "_" + Method.__str__()
    #     return Output


start_time = timeit.default_timer()
# for x in xrange(10):
#     R = Experiment(0,3,0,1)
R = Experiment(0,3,8,1)
# R = Experiment(3,3,3,1)
# R = Experiment(4,4,None,1)
# R = [(b,c,Experiment(0,3,b,c)) for b in xrange(10) for c in range(1,2)]
elapsed = timeit.default_timer() - start_time
print "Tiempo: " + elapsed.__str__()
# Out_BoW = M_BoW.show_info("Matrix_NN_BoW.jpg",True)
Out_HC = M_HC.show_info("Matrix_NN_HC.jpg",True)
# Out = M_DL.show_info("Matrix_DL_NN.jpg", True)
# joblib.dump(R,"Experiment_Combinated_Both_v2.pkl",compress=9)