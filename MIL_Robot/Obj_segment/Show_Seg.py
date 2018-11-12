from skimage.segmentation import slic
import numpy as np
import Rect
import math

class Show_Seg:

    def __init__(self):
        self.add_value = 50

    def obtain_patch(self,Image,Depth,dep,center,angle):
        def distance(p1,p2):
            x2 = p2[0]
            x1 = p1[0]
            y2 = p2[1]
            y1 = p1[1]
            dist = math.hypot(x2 - x1, y2 - y1)
            return dist

        segments_fz = slic(Image, n_segments=250, compactness=10, sigma=5)
        segments_fz+=1
        segments_ids = np.unique(segments_fz)

        # centers
        centers = np.array([np.mean(np.nonzero(segments_fz == i), axis=1) for i in segments_ids])
        #Calculate the new patch using the angle and the center
        c1 = center[1] - self.add_value
        c2 = center[1] + self.add_value
        r1 = center[0] - self.add_value
        r2 = center[0] + self.add_value
        r1 = max(1,min(r1,459))
        r2 = max(20,min(r2,479))
        c1 = max(1,min(c1,619))
        c2 = max(20,min(c2,639))
        U = np.unique(segments_fz[ r1 : r2 , c1 : c2 ],return_counts=True)
        centers = [cntr for cntr in centers]
        l= [distance(centers[u],center) for u in U[0]]
        SupeR = [U[0][i] for i in xrange(len(U[0])) if U[1][i] > 10 and l[i] < 200]
        for i in xrange(segments_fz.shape[0]):
            for j in xrange(segments_fz.shape[1]):
                if segments_fz[i][j] not in SupeR:
                    segments_fz[i][j]= 0
        non_empty_columns = np.where(segments_fz.max(axis=0) > 0)[0]
        non_empty_rows = np.where(segments_fz.max(axis=1) > 0)[0]
        cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))

        patch = Image[cropBox[0]:cropBox[1] + 1, cropBox[2]:cropBox[3] + 1]
        patch_d = Depth[cropBox[0]:cropBox[1] + 1, cropBox[2]:cropBox[3] + 1]

        R = Rect.Rect(Rect.Point(c1, r1), Rect.Point(c2, r2))
        C = R.center()
        #return the patch and the rect
        return patch,patch_d,R,C