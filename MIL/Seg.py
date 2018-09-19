import cv2
import numpy as np
from Obj_segment.Maskrcnn import *

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def prin():
	for i in list(enumerate(class_names)):
		print(" "+i[0].__str__()+" "+i[1])

def show_p(im,b):
	(x0, y0, w, h) = b
	# print(b)
	x1, y1 = int(x0 + w), int(y0 + h)
	x0, y0 = int(x0), int(y0)
	i_sub = im[y0:y1,x0:x1,:]
	prin()
	# print(i_sub.shape)
	cv2.imshow("IM",i_sub)
	cv2.waitKey()
	cat =raw_input("Category: ")
	return int(cat)

mask = maskrcnn()

img = cv2.imread("/home/pazagra/T.jpg")
print(img.shape)
l,kp = mask.maskrcnn(img)
print len(class_names)
seg = []
i = 0
for obj in l:
	if obj[2]*obj[3] > 0.8*img.shape[0]*img.shape[1]:
		continue
	i+=1
	cat =obj[4]
	print cat
	current_object = {
	"category_id": cat,
	"bbox": np.array(obj).tolist(),
	"category": class_names[cat],
	"segment": [],
	"id" : i,
	"area": 0
	}
	seg.append(current_object)
print(len(seg))
# np.save("/home/iglu/Seg2.npy",np.array(seg))


