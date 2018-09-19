import numpy as np
import cv2

class Video_saver:
    def __init__(self,name):
        self.name= name

    def Save_video(self,List_Top,List_Front,Candidates_Top,Candidates_Front,Line_Front,Line_Top):

        video = cv2.VideoWriter('Videos/'+self.name+'.avi',cv2.cv.CV_FOURCC('D', 'I', 'V', 'X'), 25.0, (640, 480))
        for L_F,L_T,C_T,C_F,Line_F,Line_T in List_Front,List_Top,Candidates_Top,Candidates_Front,Line_Front,Line_Top:
            if Line_T is None:
                No_L_Top = False
            else:
                No_L_Top = True
                P_ini_L_T, P_fin_L_T = Line_T
            P_ini_L_F, P_fin_L_F = Line_F
            video.write(Image_Completa)


        video.release()