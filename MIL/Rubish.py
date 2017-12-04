import cPickle
import gzip
import cv2
Data = cPickle.load(gzip.open("Test.pgz",'r'))
for Scena in Data:
    Im = Scena.Values["Canvas"]
    if "Hand_Pos" in Scena.Values.keys():
        x,y = Scena.Values["Hand_Pos"]
        cv2.circle(Im,(y,x),50,(255,255,126),4)
    cv2.imshow("Skel",Im)
    cv2.waitKey(41)