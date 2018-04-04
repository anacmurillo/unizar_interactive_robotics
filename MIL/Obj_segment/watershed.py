# import the necessary packages
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.segmentation import slic, felzenszwalb
from scipy import ndimage
import Obj_segment
import numpy as np
import cv2
from Obj_segment.Candidate import Candidate
homo = np.array(
            [[9.94775973e-01, -4.09621743e-01, -4.37893262e+01], [-2.06444142e-02, 2.43247181e-02, 1.94859521e+02],
             [-1.13045909e-04, -1.41217334e-03, 1.00000000e+00]])

def Homo_get(x, y, inverted=False):
    p1 = [float(x), float(y), 1.0]
    p1 = np.array(p1)
    if inverted:
        r = np.dot(p1, np.linalg.inv(homo))
    else:
        r = np.dot(homo, p1)
    r = r / r[2]
    return (r[0],r[1])


def Sp_Water(front,img,dim,top):
    candidate= []
    image = img.copy()

    labels = felzenszwalb(image, scale=150, sigma=2.5, min_size=100)
    labels += 1

    t = np.unique(labels, return_counts=True)
    for i in xrange(len(t[0])):
        if t[1][i] < 150:
            labels[labels == t[0][i]] = 1
        elif t[1][i] == max(t[1]):
            labels[labels == t[0][i]] = 1

    image[labels == 1] = (0, 0, 0)

    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20,
                              labels=thresh)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    # loop over the unique labels returned by the Watershed
    # algorithm
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue

        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)

        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        if r <= 20 or r >= 60:
            continue
        x = int(x+dim[0])
        y = int(y+dim[1])
        p1 = Homo_get(x-r,y-r)
        p2 =Homo_get(x+r,y+r)

        # print x.__str__()+" "+y.__str__()+" "+r.__str__()
        # print p1
        # print p2
        # cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        # cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        P1 = Obj_segment.Rect.Point( x-int(r),y-int(r))
        P2 = Obj_segment.Rect.Point(x+int(r),y+int(r))
        Rec_top = Obj_segment.Rect.Rect(P1, P2)

        P1 = Obj_segment.Rect.Point(max(min(int(p1[0]),639),0),max(min(int(p1[1]),480),0) )
        P2 = Obj_segment.Rect.Point(max(min(int(p2[0]),639),0),max(min(int(p2[1]),480),0))
        rec = Obj_segment.Rect.Rect(P1, P2)

        Pos_top = Rec_top.center()
        Pos_front = rec.center()
        # print rec.top
        # print rec.bottom
        Patch = front[rec.top:rec.bottom, rec.left:rec.right]
        Patch_T = top[Rec_top.top:Rec_top.bottom, Rec_top.left:Rec_top.right]

        # print Patch.shape
        Patch_d = None

        C = Candidate(Rec_top, Pos_top, rec, Pos_front, Patch,Patch_T, "WaterShed")
        candidate.append(C)
    return candidate,image
