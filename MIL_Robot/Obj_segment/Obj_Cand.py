from sklearn.cluster import MiniBatchKMeans
from Obj_segment.Maskrcnn import *
import Obj_segment.Rect
import Candidate
import watershed

class Object_Cand():

    def __init__(self,homo=None, mask= False):
        if homo is None:
            self.homo = np.array([[ 1.45511656e+00 ,-3.14859150e-01, -8.88801942e+01],
     [ 1.67450370e-02,  6.50717885e-01,  8.25769895e+01],
     [ 2.04361850e-05 ,-9.68072329e-04 , 1.00000000e+00]])
        else:
            self.homo = homo
        watershed.homo = self.homo
        self.centro = (0, 0)
        self.angulo = 0
        self.dim = None
        if mask:
            self.Mask = maskrcnn()

    def homo_get(self,x,y,inverted=False):
        p1 = [float(x), float(y), 1.0]
        p1 = np.array(p1)
        if inverted:
            r = np.dot(p1,np.linalg.inv(self.homo))
        else:
            r = np.dot(self.homo, p1)
        r = r / r[2]
        return (r[0], r[1])

    def top_to_front(self,x,y,x2,y2,im_front,im_top,dep_top):
        p1 = self.homo_get(x, y)
        p2 = self.homo_get(x2, y2)

        P1 = Obj_segment.Rect.Point(x, y)
        P2 = Obj_segment.Rect.Point(x2, y2)
        Rec_top = Obj_segment.Rect.Rect(P1, P2)

        P1 = Obj_segment.Rect.Point(max(min(int(p1[0]), 639), 0), max(min(int(p1[1]), 480), 0))
        P2 = Obj_segment.Rect.Point(max(min(int(p2[0]), 639), 0), max(min(int(p2[1]), 480), 0))
        rec = Obj_segment.Rect.Rect(P1, P2)

        Pos_top = Rec_top.center()
        Pos_front = rec.center()

        Patch = im_front[rec.top:rec.bottom, rec.left:rec.right]
        Patch_T = im_top[Rec_top.top:Rec_top.bottom, Rec_top.left:Rec_top.right]
        Patch_d = dep_top[Rec_top.top:Rec_top.bottom, Rec_top.left:Rec_top.right]
        C = Candidate.Candidate(Rec_top, Pos_top, rec, Pos_front, Patch,Patch_T, Patch_d)

        return C


    def get_candidate(self,FM):
        num = 0
        Cand = []
        points_top = []
        f = []
        Objects = []
        for S in FM.Images:
            dim, img = self.get_dim(cv2.imread(S.RGB_top), S.Mask_top)
            if dim is None or img is None:
                continue
            S.Values["dim"] = dim
            if num < 1:
                num += 1
                if self.Mask is not None:
                    lista,_ = self.Mask.maskrcnn(img)
                    for l in lista:
                        c1 = dim[0] + l[0]
                        c2 = dim[0] + l[0] +l[2]
                        r1 = dim[1] + l[1]
                        r2 = dim[1] + l[1] +l[3]
                        C = self.top_to_front(c1, r1, c2, r2, cv2.imread(S.RGB_front),cv2.imread(S.RGB_top),np.load(S.Depth_top))
                        if C.size_top == 0 or C.size_front == 0:
                            continue
                        Objects.append(C)
                T, img = watershed.Sp_Water(cv2.imread(S.RGB_front), img, dim,cv2.imread(S.RGB_top))
                for c in T:
                    if c.size_top == 0 or c.size_front == 0:
                        continue
                    Cand.append(c)
                    p1 = c.BB_top.center()
                    points_top.append(p1)
                top1 = cv2.imread(S.RGB_top)
                top2 = cv2.imread(S.RGB_front)
                color = (255,0,0)
                for ref in Objects:
                    p1, p2 = ref.BB_top.two_point()
                    cv2.rectangle(top1, p1, p2, color, 2)
                    p1, p2 = ref.BB_front.two_point()
                    cv2.rectangle(top2, p1, p2, color, 2)
                for ref in Cand:
                    p1, p2 = ref.BB_top.two_point()
                    cv2.rectangle(top1, p1, p2, color, 2)
                    p1, p2 = ref.BB_front.two_point()
                    cv2.rectangle(top2, p1, p2, color, 2)
                cv2.imshow('AS1',top1)
                cv2.imshow('AS2',top2)
                cv2.waitKey(3000)
        if len(points_top) < 12:
            for c in Cand:
                Objects.append(c)
            return Objects
        kmeans = MiniBatchKMeans(n_clusters=12).fit(np.array(points_top))
        for num in xrange(len(kmeans.cluster_centers_)):
            indx = np.where(kmeans.labels_ == num)[0]
            if len(indx) == 0:
                continue
            center = kmeans.cluster_centers_[num]
            ind = indx[0]
            min_d = np.linalg.norm(center - np.array(Cand[ind].BB_top.center()))
            for nma in xrange(1, len(indx)):
                dist = np.linalg.norm(center - np.array(Cand[indx[nma]].BB_top.center()))
                if dist < min_d:
                    ind = indx[nma]
                    min_d = dist
            Objects.append(Cand[ind])
        top = cv2.imread(FM.Images[0].RGB_top)
        color=(255,0,0)
        for ref in Objects:
            p1, p2 = ref.BB_top.two_point()
            cv2.rectangle(top, p1, p2, color, 2)
        return Objects

    def get_dim(self,Img,Mask1):
        #Process the Mask of the top camera
        Mask_1 = cv2.imread(Mask1).copy()
        kernel = np.ones((7, 7), np.uint8)
        Mask_1 = cv2.dilate(Mask_1, kernel, 1)
        kernel = np.ones((4, 4), np.uint8)
        Mask_1 = cv2.erode(Mask_1, kernel, 1)
        edged = cv2.Canny(Mask_1,1,240)
        i,j = np.where(edged == 255)
        y=int(np.mean(i))
        dim = (0,y,640,480-y)
        img = Img[y:480,:,:]
        return dim,img


