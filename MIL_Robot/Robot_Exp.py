from Interaction import *
from Scene.FMI import *
from Obj_segment import *
from Descriptors.Compute_Descriptors import *
from Incremental_learning.NN_modded import *
from Incremental_learning.Model import *
import DataExtractor
import os
import rospy
import pygame
import Robot_Controller
from gtts import gTTS
import cv_bridge
from sensor_msgs.msg import (
    Image,
)
import speech_recognition as sr
import Config

Descript = "HC"
if os.path.exists('Kinects.npy') and os.path.exists('KRobot.npy'):
    Homo = np.load('Kinects.npy')
    Robot_Controller.h = np.load('KRobot.npy')
elif os.path.exists('Kinects.npy') and not os.path.exists('KRobot.npy'):
    Homo = np.load('Kinects.npy')
    _, Robot_Controller.h = Robot_Controller.standar_calibration(True)
else:
    Homo, Robot_Controller.h = Robot_Controller.standar_calibration()
I_front = Interaction_Multiple.Mutiple_Interaction()
Obj_candidate = Obj_Cand.Object_Cand(Homo,mask=Config.USE_MASKRCNN)
D = D_calculator()


def send_image(path,patches=None):

    img = cv2.imread("Images/"+path)
    if patches!= None:
        pat = []
        if len(patches)==1:
            pat.append(patches[0])
        else:
            for p in patches:
                p=cv2.resize(p,(200,200))
                pat.append(p)
        if len(pat)==2:
            im = np.hstack(pat)
            im = cv2.resize(im,(400,200))
            img[195:395,441:841] = im
        elif len(pat)==1:
            im = cv2.resize(pat[0],(400,400))
            img[195:595, 441:841] = im
        else:
            im1 = np.hstack(pat[0:len(pat)/2])
            im1 = cv2.resize(im1, (400, 200))
            im2 = np.hstack(pat[len(pat)/2:])
            im2 = cv2.resize(im2, (400, 200))
            im=np.vstack((im1,im2))
            img[195:595, 441:841] = im
    msg = cv_bridge.CvBridge().cv2_to_imgmsg(img, encoding="bgr8")
    pub = rospy.Publisher('/display', Image, latch=True)
    pub.publish(msg)
    img = cv2.resize(img, (1024, 768))
    img = cv2.flip(img, -1)
    msg = cv_bridge.CvBridge().cv2_to_imgmsg(img, encoding="bgr8")
    pub = rospy.Publisher('/robot/xdisplay', Image, latch=True)
    pub.publish(msg)
    # Sleep to allow for image to be published.
    rospy.sleep(1)


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
    if Config.USE_SPEECH:
        tts = gTTS(text=msg, lang='en')
        tts.save("audio.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load("audio.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() == True:
            continue
        pygame.mixer.music.stop()


def audio_chose():
    if Config.USE_VOICE_RECOG:
        r = sr.Recognizer()
        m = sr.Microphone()

        play_audio("A moment of silence, please...")
        with m as source:
            r.adjust_for_ambient_noise(source)
        print("Set minimum energy threshold to {}".format(r.energy_threshold))
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
                    return "t"
                elif "search" in txt and "all" in txt:
                    return "a"
                elif "search" in txt:
                    return "s"
                elif "leave" in txt:
                    return "l"
            except sr.UnknownValueError:
                print("Oops! Didn't catch that")
            except sr.RequestError as e:
                print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
    else:
        return raw_input("")


def choose():
    play_audio("Hi There! Welcome")
    print "Hi There! Welcome"
    send_image("Hello.jpg")
    Incremental = NN_new(0.2,20)
    if os.path.exists("Incremental_NN.pkl"):
        Incremental.load("Incremental_NN.pkl")
    else:
        send_image("Prior.jpg")
        play_audio("I haven't found prior knowledge, so let's start fresh")
        print "I haven't found prior knowledge, so let's start fresh"
    print "So, What you wanna do?"
    play_audio("So, What you wanna do?")
    send_image("Chose.jpg")
    play_audio("type t for teach, s for search ,a for all objects and l for leave and press enter")
    print("(Teach(t),Search(s),All(a),Leave(l)) & Enter")
    while 1:
        c = audio_chose()
        if c ==  't':
            teach(Incremental)
            print "What is next?"
            play_audio("What is next?")
        elif c == 's':
            search(Incremental)
            print "What is next?"
            play_audio("What is next?")
        elif c == 'a':
            search_all(Incremental)
            print "What is next?"
            play_audio("What is next?")
        elif c == 'l':
            break
        send_image("Chose.jpg")
    send_image("Bye.jpg")
    print "See you soon"
    play_audio("See you soon")
    Incremental.dump("Incremental_NN.pkl")


def move_hand_robot(x,y):
    Robot_Controller.move_hand(x, y)

def obtain_images(FM):
    cv = DataExtractor.DataExtractor("", 3)
    cv.start()
    print "start"
    rospy.sleep(5)
    print "end"
    FM.Images= cv.finish()

global p
def draw_circle(event,x,y,flags,param):
    global p
    if event == cv2.EVENT_LBUTTONUP:
        p.append([x,y])

def search_all(Incremental):
    global p
    p=[]
    play_audio("Let me see, one sec.")
    send_image('Obj1.jpg')
    FM = Full_Movie('live', 'live', '')
    obtain_images(FM)
    T = Obj_candidate.get_candidate(FM)
    FM.Candidate_patch = T
    for candidate in FM.Candidate_patch:
        candidate.Descriptors = D.calculate_D(candidate.patch, candidate.patch_d, [Descript])
        if candidate.patch_T is not None:
            candidate.Descriptors_T = D.calculate_D(candidate.patch_T, candidate.patch_d,
                                                    [Descript])
    top = cv2.imread(FM.Images[0].RGB_top)
    for candidate in FM.Candidate_patch:
        if Descript not in candidate.Descriptors.keys():
            continue
        M = Model('hist', candidate.Descriptors[Descript], None, None)
        out,dt = Incremental.ntest(M, 2)
        out = [set_name(o) for o in out]
        p1, p2 = candidate.BB_top.two_point()
        cv2.rectangle(top, p1, p2, (250,200,0), 2)
        cv2.putText(top, out[0], p1, cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
        cv2.putText(top, out[1], p2, cv2.FONT_ITALIC, 0.5, (125, 0, 0),1)
    send_image("Scan.jpg",[top])
    cv2.destroyAllWindows()
    play_audio("I have scanned all objects in the Table")
    rospy.sleep(2)


def search(Incremental):
    global p
    p=[]
    play_audio("Let me see, one sec.")
    send_image('Obj1.jpg')
    FM = Full_Movie('live', 'live', '')
    obtain_images(FM)
    T = Obj_candidate.get_candidate(FM)
    send_image("Scan.jpg")
    play_audio("I have scanned all objects in the Table")
    FM.Candidate_patch = T
    for candidate in FM.Candidate_patch:
        candidate.Descriptors = D.calculate_D(candidate.patch, candidate.patch_d, [Descript])
        if candidate.patch_T is not None:
            candidate.Descriptors_T = D.calculate_D(candidate.patch_T, candidate.patch_d,
                                                    [Descript])
    send_image("Search.jpg")
    play_audio("What object are you looking for?")
    print("What object are you looking for?")
    cv2.destroyAllWindows()
    c = raw_input("(Object):")
    d = 9999999999
    top = cv2.imread(FM.Images[0].RGB_top)
    cand = None
    for candidate in FM.Candidate_patch:
        M = Model('hist', candidate.Descriptors[Descript], None, None)
        out,dt = Incremental.ntest(M, 2)
        out = [set_name(o) for o in out]
        if len(out)==3:
            if out[2] == out[1]:
                label = out[1]
                dist = dt[1]
            else:
                label = out[0]
                dist = dt[0]
        else:
            label = out[0]
            dist = dt[0]
        p1, p2 = candidate.BB_top.two_point()
        cv2.rectangle(top, p1, p2, (250,200,0), 2)
        cv2.putText(top, out[0], p1, cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
        cv2.putText(top, out[1], p2, cv2.FONT_ITALIC, 0.5, (125, 0, 0), 1)
        if label == c and dist < d:
            cand = candidate
            d = dist
    cv2.imshow("Full",top)
    cv2.waitKey(5000)
    if cand is not None:
        send_image("Found.jpg")
        play_audio("I found it!")
        print("I found it!")
        x,y = cand.BB_top.center()
        move_hand_robot(x,y)
    else:
        send_image("NotFound.jpg")
        play_audio("Sorry, i can't find it.")
        print("Sorry, i can't find it.")
    cv2.destroyAllWindows()



def audio(cv=None):
    if Config.USE_VOICE_RECOG:
        r = sr.Recognizer()
        m = sr.Microphone()

        play_audio("A moment of silence, please...")
        with m as source:
            r.adjust_for_ambient_noise(source)
        print("Set minimum energy threshold to {}".format(r.energy_threshold))
        Label =""
        while True:
            play_audio("You can start")
            if cv is not None:
                cv.start()
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
    else:
        return raw_input("")


def Obtain_FM(FM):
    send_image("Teach1.jpg")
    play_audio("Okay, Teach me one object")
    cv2.destroyAllWindows()
    cv = DataExtractor.DataExtractor("", 8)
    play_audio("Press enter to start and enter to finish")
    if Config.USE_VOICE_RECOG:
        L = audio(cv)
    else:
        raw_input("")
        L = cv.start()
        raw_input("")
    FM.Images= cv.finish()
    FM.Speech = L.split(" ")
    print "Number of Images taken: "+len(FM.Images).__str__()

def teach(Incremental):
    FM = Full_Movie('live', 'live', '')
    Obtain_FM(FM)
    I_front.__del__()
    send_image("Process.jpg")
    play_audio("I will process the info, please, wait")
    print("I will process the info, please, wait")
    I_front.Calculate_Interaction(FM)
    if FM.Values["Class_N"] == "Point":
        send_image("Point1.jpg")
        play_audio("You were pointing an object.Is that right?")
        cv2.destroyAllWindows()
        c = raw_input("You were pointing an object. (y/n):")
        if c == 'n':
            play_audio("I'm sorry")
            FM.Values["Class_N"]= 'Show'
    else:
        send_image("Show1.jpg")
        play_audio("You were showing me an object.Is that right?")
        cv2.destroyAllWindows()
        c = raw_input("You were showing me an object. (y/n):")
        if c == 'n':
            play_audio("I'm sorry")
            FM.Values["Class_N"] = 'Point'
    send_image("Obj1.jpg")
    if FM.Values["Class_N"] == "Point":
        FM1 = Full_Movie('live', 'live', '')
        obtain_images(FM1)
        T = Obj_candidate.get_candidate(FM1)
        for S in FM.Images:
            S.addObj(T)
        FM.Objects = T
    else:
        FM.Objects = []
    Search_Reference.Search_Ref(FM)
    pat = []
    if FM.Candidate_patch == []:
        play_audio("Sorry, i coudn't see the object...")
        send_image("NotFound.jpg")
        return 0
    for candidate in FM.Candidate_patch:
        pat.append(candidate.patch)
        if candidate.patch_T is not None:
            pat.append(candidate.patch_T)
    cv2.destroyAllWindows()
    send_image("Obj2.jpg",pat)
    play_audio("So, what is that object?")
    print("So, what is that object?")
    Label = raw_input("Name: ")
    if Label== 'None':
        play_audio("I will ignore it.")
        return 0
    for candidate in FM.Candidate_patch:
        candidate.Label = Label
        candidate.Descriptors = D.calculate_D(candidate.patch, candidate.patch_d, [Descript])
        M = Model('hist', candidate.Descriptors[Descript], None, None)
        Incremental.train(M, candidate.Label)
        if candidate.patch_T is not None:
            candidate.Descriptors_T = D.calculate_D(candidate.patch_T, candidate.patch_d,
                                                    [Descript])
            M = Model('hist', candidate.Descriptors_T[Descript], None, None)
            Incremental.train(M, candidate.Label)
    cv2.destroyAllWindows()
    Incremental.dump("Incremental_NN.pkl")
    play_audio("I have added to my knowledge")
    print("Learned Finished")

rospy.init_node('talker', anonymous=True)
choose()