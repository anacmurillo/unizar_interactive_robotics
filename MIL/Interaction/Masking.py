import numpy as np
import cv2

class Masking:
    def __init__(self):
        pass

    def Mask(self,Mask,dep):
        # USER EXTRACTION USING DEPTH
        ret, mask = cv2.threshold(dep, 1.7, 1, cv2.THRESH_BINARY_INV)
        mask = np.uint8(mask)
        kernel = np.ones((5, 5), np.uint8)
        Mask = Mask[:, :, 0]
        Mask[Mask == 0] = 1
        Mask[Mask > 1] = 0
        Mask = cv2.morphologyEx(Mask, cv2.MORPH_CLOSE, kernel)
        Mask = Mask +mask
        Mask -=1
        return Mask