from Obj_segment import Point_Seg,Show_Seg,Candidate
import numpy as np
def Search_Ref(FM):
    Interaction = FM.Values["Class_N"]

    if Interaction == 'Point':
        P_seg = Point_Seg.Point_Seg()
        P_seg.add_candidates(FM.Objects)
        L = []
        for Im in FM.Images:
            if "Hand_Pos" not in Im.Values.keys() or "Hand_Angle" not in Im.Values.keys():
                continue
            p_ini = Im.Values["Hand_Pos"]
            angle = Im.Values["Hand_Angle"]
            if p_ini is None or angle is None:
                continue
            cand = P_seg.obtain_candidate(p_ini,angle,False)
            for c in cand:
                L.append(FM.Objects.index(c))
        if L == []:
            print "Error: No se ha encontrado ningun candidato"
        else:
            print np.unique(L, return_counts=True)
            T = np.unique(L, return_counts=True)[1]
            i = T.tolist().index(max(T))
            FM.Candidate_patch = [FM.Objects[i]]
    elif Interaction == 'Show':
        S_seg = Show_Seg.Show_Seg()
        altura = np.mean([scena.Values["Hand_Pos"][0] for scena in FM.Images if "Hand_Pos" in scena.Values.keys()])
        List = []
        for scena in FM.Images:
            if "Hand_Pos" not in scena.Values.keys() or "Hand_Angle" not in scena.Values.keys():
                continue
            Image = scena.RGB_front
            Depth = scena.Depth_front
            dep = scena.Values["Mask"]
            center = scena.Values["Hand_Pos"]
            angle = scena.Values["Hand_Angle"]
            if center[0] >= 0.75*altura:
                patch,rec,pos= S_seg.obtain_patch(Image,Depth,dep,center,angle)
                C = Candidate.Candidate(None, None, rec, pos, patch)
                List.append(C)
        FM.Candidate_patch = List
    else: #Speak
        pass
        #Sp_seg = Speak_Seg.Speak_Seg()