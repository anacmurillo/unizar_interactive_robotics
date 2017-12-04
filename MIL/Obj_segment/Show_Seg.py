import math
import cv2
from skimage.segmentation import slic,mark_boundaries
import numpy as np
import Rect
from skimage import color

class Show_Seg:

    def __init__(self):
        self.add_value = 50

    def obtain_patch(self,Image,Depth,dep,center,angle):

        #Mask the image
        ret, mask = cv2.threshold(Depth, 1.7, 1, cv2.THRESH_BINARY_INV)
        mask = np.uint8(mask)
        dep = cv2.bitwise_not(dep)
        mask = cv2.bitwise_and(mask,dep)
        masked_data = cv2.bitwise_and(Image, Image, mask=mask)

        #Calculate the new patch using the angle and the center
        c1 = center[1] - self.add_value
        c2 = center[1] + self.add_value
        r1 = center[0] - self.add_value
        r2 = center[0] + self.add_value
        if 45 > angle > -45:
            r2 += self.add_value
        elif 45 < angle < 90:
            c2 += self.add_value
        elif angle > 90 or angle <-90:
            r1 -=self.add_value
        else:
            c1 -= self.add_value

        r1 = max(0,min(r1,639))
        r2 = max(0,min(r2,639))
        c1 = max(0,min(c1,479))
        c2 = max(0,min(c2,479))

        masked_data[0:c1,:]=0
        masked_data[c2:480, :] = 0
        masked_data[:,0:r1] = 0
        masked_data[:,r2:640] = 0

        R = Rect.Rect((r1, c1), (r2, c2))
        C = R.center()
        #return the patch and the rect
        return masked_data[ c1 : c2 , r1 : r2 ],R,C