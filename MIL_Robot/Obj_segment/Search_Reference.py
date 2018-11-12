from Obj_segment import Point_Seg,Show_Seg,Candidate,Rect
import numpy as np
import cv2

def Search_Ref(FM):
    Interaction = FM.Values["Class_N"]
    if Interaction == 'Point':
        P_seg = Point_Seg.Point_Seg()
        P_seg.add_candidates(FM.Objects)
        L = []
        for Im in FM.Images:
            if "Hand_Pos" not in Im.Values.keys() or "Hand_Angle" not in Im.Values.keys():
                continue
            p_ini = (Im.Values["Hand_Pos"][1], Im.Values["Hand_Pos"][0])
            angle = Im.Values["Hand_Angle"]
            if p_ini is None or angle is None:
                continue
            cand = P_seg.obtain_candidate(p_ini,angle,False,cv2.imread(Im.RGB_front))
            Im.Values["Candidates"] = cand
            for c in cand:
                L.append(FM.Objects.index(c))
        if L == []:
            FM.Candidate_patch = L
            print "Error: No se ha encontrado ningun candidato"
        else:
            T = np.unique(L, return_counts=True)
            Scores = []
            for n in xrange(len(T[0])):
                score =T[1][n]/FM.Objects[T[0][n]].BB_front.d_diag(5)
                Scores.append((T[0][n],score))
            Scores = sorted(Scores,key = lambda s: s[1],reverse= True)
            n =0
            i = Scores[0][0]
            while Rect.contains(FM.Objects[i].BB_front,p_ini):
                n+=1
                i = Scores[n][0]
                if i>=len(FM.Objects):
                    i=0
                    break
            FM.Candidate_patch = [FM.Objects[i]]
    elif Interaction == 'Show':
        S_seg = Show_Seg.Show_Seg()
        maxi = np.max([scena.Values["Hand_Pos"][0] for scena in FM.Images if "Hand_Pos" in scena.Values.keys()])
        min = np.min([scena.Values["Hand_Pos"][0] for scena in FM.Images if "Hand_Pos" in scena.Values.keys()])
        altura = maxi-((maxi-min)*0.8)
        List = []
        for scena in FM.Images:
            if not ("Hand_Pos" in scena.Values.keys() or "Hand_Angle" in scena.Values.keys()):
                continue
            Image = cv2.imread(scena.RGB_front)
            Depth = np.load(scena.Depth_front)
            dep = scena.Values["Mask"]
            center = scena.Values["Hand_Pos"]
            angle = scena.Values["Hand_Angle"]
            # print center[0]
            if center[0] <= altura:
                patch,patch_d,rec,pos= S_seg.obtain_patch(Image,Depth,dep,center,angle)
                cv2.rectangle(Image,(rec.left,rec.top),(rec.right,rec.bottom),(255,255,0))
                cv2.imshow("Patch",patch)
                cv2.imshow("Imagen",Image)
                cv2.waitKey(40)
                C = Candidate.Candidate(None, None, rec, pos, patch,None,patch_d)
                scena.Values["Candidates"] = [C]
                List.append(C)
        FM.Candidate_patch = List