CAFFE_MODELS = "/home/pazagra/MIL/Interaction/model_hand/"
import sys
import cv2 as cv
import math
from caffe2.proto import caffe2_pb2
import numpy as np
from caffe2.python import core, workspace
import os
import matplotlib.pyplot as pyplot
from scipy.ndimage.filters import gaussian_filter
import time


def process_frames(frame, boxsize=368, scales=[1]):
    base_net_res = None
    imagesForNet = []
    imagesOrig = []
    for idx, scale in enumerate(scales):
        # Calculate net resolution (width, height)
        if idx == 0:
            net_res = (16 * int((boxsize * frame.shape[1] / float(frame.shape[0]) / 16) + 0.5), boxsize)
            base_net_res = net_res
        else:
            net_res = ((min(base_net_res[0], max(1, int((base_net_res[0] * scale) + 0.5) / 16 * 16))),
                       (min(base_net_res[1], max(1, int((base_net_res[1] * scale) + 0.5) / 16 * 16))))
        input_res = [frame.shape[1], frame.shape[0]]
        scale_factor = min((net_res[0] - 1) / float(input_res[0] - 1), (net_res[1] - 1) / float(input_res[1] - 1))
        print scale_factor
        warp_matrix = np.array([[scale_factor, 0, 0],
                                [0, scale_factor, 0]])
        print warp_matrix
        if scale_factor != 1:
            imageForNet = cv.warpAffine(frame, warp_matrix, net_res,
                                         flags=(cv.INTER_AREA if scale_factor < 1. else cv.INTER_CUBIC),
                                         borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
        else:
            imageForNet = frame.copy()

        imageOrig = imageForNet.copy()
        imageForNet = imageForNet.astype(float)
        imageForNet = imageForNet / 256. - 0.5
        imageForNet = np.transpose(imageForNet, (2, 0, 1))

        imagesForNet.append(imageForNet)
        imagesOrig.append(imageOrig)


    return imagesForNet, imagesOrig,warp_matrix

def poseFromHM(image, hm, ratios=[1]):
        """
        Pose From Heatmap: Takes in an image, computed heatmaps, and require scales and computes pose
        Parameters
        ----------
        image : color image of type ndarray
        hm : heatmap of type ndarray with heatmaps and part affinity fields
        ratios : scaling ration if needed to fuse multiple scales
        Returns
        -------
        array: ndarray of human 2D poses [People * BodyPart * XYConfidence]
        displayImage : image for visualization
        """
        if len(ratios) != len(hm):
            raise Exception("Ratio shape mismatch")
        # Find largest
        # hm_combine = np.zeros(shape=(1, hm.shape[1], hm.shape[2], hm.shape[3]),dtype=np.float32)
        # i=0
        # for h in hm:
        #    hm_combine[i,:,0:h.shape[2],0:h.shape[3]] = h
        #    i+=1
        # hm = hm_combine

        ratios = np.array(ratios,dtype=np.float32)

        shape = image.shape
        displayImage = np.zeros(shape=(image.shape),dtype=np.uint8)
        size = np.zeros(shape=(4),dtype=np.int32)
        size[0] = hm.shape[0]
        size[1] = hm.shape[1]
        size[2] = hm.shape[2]
        size[3] = hm.shape[3]

        # self._libop.poseFromHeatmap(self.op, image, shape[0], shape[1], displayImage, hm, size, ratios)
        array = np.zeros(shape=(size[0],size[1],size[2]),dtype=np.float32)
        # self._libop.getOutputs(self.op, array)
        # return array, displayImage
        return size,shape,ratios,displayImage

def nanargmax(a):
    idx = np.argmax(a, axis=None)
    multi_idx = np.unravel_index(idx, a.shape)
    if np.isnan(a[multi_idx]):
        nan_count = np.sum(np.isnan(a))
        # In numpy < 1.8 use idx = np.argsort(a, axis=None)[-nan_count-1]
        idx = np.argpartition(a, -nan_count-1, axis=None)[-nan_count-1]
        multi_idx = np.unravel_index(idx, a.shape)
    return multi_idx

oriImg = cv.imread("/home/pazagra/Hand.jpg")
INIT_NET = os.path.join(CAFFE_MODELS, "init_net.pb")
print 'INIT_NET = ', INIT_NET
PREDICT_NET = os.path.join(CAFFE_MODELS, "predict_net.pb")
print 'PREDICT_NET = ', PREDICT_NET
device_opts = core.DeviceOption(caffe2_pb2.CUDA, 0)
init_def = caffe2_pb2.NetDef()
with open(INIT_NET, "rb") as f:
    init_def.ParseFromString(f.read())
    init_def.device_option.CopyFrom(device_opts)
    workspace.RunNetOnce(init_def.SerializeToString())

net_def = caffe2_pb2.NetDef()
with open(PREDICT_NET,"rb") as f:
    net_def.ParseFromString(f.read())
    net_def.device_option.CopyFrom(device_opts)
    workspace.CreateNet(net_def.SerializeToString(), True)
# from caffe2.python import net_drawer
# from IPython import display
# graph = net_drawer.GetPydotGraph(net_def, rankdir="LR")
# graph.write_png("T.png")
data = np.zeros((1, 3, 368, 368))
imageToTest_padded,im,WM = process_frames(oriImg)
imageToTest_padded = imageToTest_padded[0]
data.resize(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
data = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3,0,1,2)) / 256 - 0.5
print data.shape
workspace.FeedBlob('image', data, device_option=device_opts)
workspace.RunNet('')
output = workspace.FetchBlob('net_output')
print output.shape
for i in xrange(22):
    t1 = output[0,i,:,:].copy()
    oriImg = cv.resize(oriImg,(46,46), interpolation=cv.INTER_CUBIC)
    y,x = nanargmax(t1)
    print y
    print x
    cv.circle(oriImg,(x,y),1,(255,0,0),2)
    # pyplot.imshow(t1)
oriImg = cv.resize(oriImg,(80,80), interpolation=cv.INTER_CUBIC)
cv.imshow("output", oriImg)
cv.waitKey()