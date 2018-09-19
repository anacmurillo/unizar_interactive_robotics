import cv2
import numpy as np
from Interaction import *
import sys
import time
from Scene.Scene import *
from Scene.FMI import *
from Obj_segment import *
from Descriptors.Compute_Descriptors import *
from Incremental_learning.NN_modded import *
from Obj_segment.watershed import Sp_Water
import cPickle
import gzip
import subprocess
import json
import ExtractRosbag
import os
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
import pygame
import tf
import yaml
import shutil
from gtts import gTTS

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

def play_audio(msg):
    tts = gTTS(text=msg, lang='en')
    tts.save("audio.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load("audio.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue
    pygame.mixer.music.load("hold.mp3")


def table_mask(RGB,Depth):
    sys.stdout.flush()
    cv2.imwrite("RGB.jpg",RGB)
    with open('Depth.yaml', 'w') as f:
        f.write("%YAML:1.0")
        yaml.dump({"Depth": Depth}, f)
    prg = "./pcl_visualizer_demo " + "RGB.jpg" + " " + "Depth.yaml"
    os.system(prg)
    os.remove("Depth.yaml")
    os.remove("RGB.jpg")
    Mask = cv2.imread("Mask.jpg")
    return Mask

def audio_chose():
    import speech_recognition as sr

    r = sr.Recognizer()
    m = sr.Microphone()

    play_audio("A moment of silence, please...")
    with m as source:
        r.adjust_for_ambient_noise(source)
    print("Set minimum energy threshold to {}".format(r.energy_threshold))
    Label =""
    play_audio("You can start")
    while True:
        with m as source:
            audio = r.listen(source)
        try:
            value = r.recognize_google(audio)
            if str is bytes:
                txt = format(value).encode("utf-8")
            else:
                txt = format(value)
            print("You said:"+txt)
            if "teach" in txt:
                return "teach"
            elif "search" in txt:
                return "search"
            elif "leave" in txt:
                return "leave"
        except sr.UnknownValueError:
            print("Oops! Didn't catch that")
        except sr.RequestError as e:
            print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))


def choose():
    # tf_listener = tf.TransformListener()
    # now = rospy.Time().now()
    # tf_listener.waitForTransform("/base", "/head_camera", now, rospy.Duration(4))
    # (trans, rot) = tf_listener.lookupTransform("/base", "/head_camera", now)
    play_audio("Hi There! Welcome")
    print "Hi There! Welcome"
    Incremental = NN_new(0.2,20)
    if os.path.exists("Incremental_NN.pkl"):
        Incremental.load("Incremental_NN.pkl")
    else:
        play_audio("I haven't found prior knowledge, so let's start fresh")
        print "I haven't found prior knowledge, so let's start fresh"
    print "So, What you wanna do?"
    play_audio("So, What you wanna do?")
    while 1:
        print("(Teach,Search,Leave)")
        c=audio_chose()
        if c == 'Teach' or c == 'teach':
            teach(Incremental)
            print "What is next?"
            play_audio("What is next?")
        elif c == 'Search' or c== 'search':
            search(Incremental,trans,rot)
            print "What is next?"
            play_audio("What is next?")
        else:
            break
    print "See you soon"
    play_audio("See you soon")
    Incremental.dump("Incremental_NN.pkl")


def move_hand_robot(x,y,z):
    pub = rospy.Publisher('Mov', String, queue_size=2)
    pub.publish(x.__str__()+";"+y.__str__()+";"+z.__str__())

def obtain_images(FM):
    cv = ExtractRosbag.cvBridgeDemo("","",0,10)
    print "A"
    rospy.sleep(4)
    print "B"
    FM.Images= cv.finish()

def search(Incremental,trans,rot):
    # talker()
    play_audio("What object are you looking for?")
    print("What object are you looking for?")
    Obj_candidate = Obj_Cand.Object_Cand()
    D = D_calculator()
    FM = Full_Movie('live', 'live', '')
    obtain_images(FM)
    T = Obj_candidate.get_candidate(FM)
    for candidate in FM.Candidate_patch:
        candidate.Label = FM.Speech[1].replace('\n', '')
        candidate.Descriptors = D.calculate_D(candidate.patch, candidate.patch_d, ["HC"])
        if candidate.patch_T is not None:
            candidate.Descriptors_T = D.calculate_D(candidate.patch_T, candidate.patch_d,
                                                    ["HC"])
    c = raw_input("(Object):")
    d = 9999999999
    cand = None
    for candidate in FM.Candidate_patch:
        M = Model.Model('hist', candidate.Descriptors["HC"], None, None)
        out,dt = Incremental.ntest(M, 3)
        # out = NN_BoW.ntest(candidate.Descriptors[d], 3)
        out = [set_name(o) for o in out]
        if out[2] == out[1]:
            label = out[1]
            dist = dt[1]
        else:
            label = out[0]
            dist = dt[0]
        if label == c and dist < d:
            cand = candidate
            d = dist

    if cand is not None:
        play_audio("I found it!")
        print("I found it!")
        x,y,z = cand.get_center(trans,rot)
        move_hand_robot(x,y,z)
    else:
        play_audio("Sorry, i can't find it.")
        print("Sorry, i can't find it.")


def audio():
    import speech_recognition as sr

    r = sr.Recognizer()
    m = sr.Microphone()

    play_audio("A moment of silence, please...")
    with m as source:
        r.adjust_for_ambient_noise(source)
    print("Set minimum energy threshold to {}".format(r.energy_threshold))
    Label =""
    while True:
        play_audio("You can start")
        with m as source:
            audio = r.listen(source)
        try:
            # recognize speech using Google Speech Recognition
            value = r.recognize_google(audio)

            # we need some special handling here to correctly print unicode characters to standard output
            if str is bytes:  # this version of Python uses bytes for strings (Python 2)
                txt = format(value).encode("utf-8")
            else:  # this version of Python uses unicode for strings (Python 3+)
                txt = format(value)
            print txt
            if "done" in txt:
                txt.replace("done","")
                Label= Label+txt
                break
            else:
                Label = txt
        except sr.UnknownValueError:
            print("Oops! Didn't catch that")
        except sr.RequestError as e:
            print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
    return Label


def Obtain_FM(FM):
    cv = ExtractRosbag.cvBridgeDemo("","",0,0)
    L = audio()
    FM.Images= cv.finish()
    FM.Speech = L

def teach(Incremental):
    # talker()
    play_audio("Okay, Teach me one object")
    I_front = Interaction_Multiple.Mutiple_Interaction()
    Obj_candidate = Obj_Cand.Object_Cand()
    D = D_calculator()
    FM = Full_Movie('live', 'live', '')
    Obtain_FM(FM)
    I_front.Calculate_Interaction(FM)
    if FM.Values["Class_N"] in ["Point", "Speak"]:
        T = Obj_candidate.get_candidate(FM)
        for S in FM.Images:
            S.addObj(T)
        FM.Objects = T
    else:
        FM.Objects = []
    Search_Reference.Search_Ref(FM)
    for candidate in FM.Candidate_patch:
        candidate.Label = FM.Speech[1].replace('\n', '')
        candidate.Descriptors = D.calculate_D(candidate.patch, candidate.patch_d, ["HC"])
        if candidate.patch_T is not None:
            candidate.Descriptors_T = D.calculate_D(candidate.patch_T, candidate.patch_d,
                                                    ["HC"])

    for candidate in FM.Candidate_patch:
        Incremental.train((candidate.Descriptors["HC"], FM.user), candidate.Label)
    play_audio("I have added to my knowledge")
    print("Learned Finished")
FM = Full_Movie('live', 'live', '')
data = Obtain_FM(FM)
print FM.Speech
exit()

choose()