import sys
import cv2
import numpy as np

class Alexnet():

    def __init__(self):
        None
        # model_file = 'Descriptors/models/bvlc_reference_caffenet.caffemodel'
        # deploy_prototxt = 'Descriptors/models/deploy.prototxt'
        # self.net = caffe.Net(deploy_prototxt, model_file, caffe.TEST)
        # self.imagemean_file = 'Descriptors/models/ilsvrc_2012_mean.npy'
        # self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        # self.transformer.set_mean('data', np.load(self.imagemean_file).mean(1).mean(1))
        # self.transformer.set_transpose('data', (2, 0, 1))
        # self.transformer.set_raw_scale('data', 255.0)

    def ComputeFc7(self,RGB,Depth):
        #
        # layer = 'fc7'
        # if layer not in self.net.blobs:
        #     return None,None
        #
        # self.net.blobs['data'].reshape(1, 3, 227, 227)
        # self.net.blobs['data'].data[...] = self.transformer.preprocess('data', RGB)
        #
        # output = self.net.forward()
        return None,None
        # return self.net.blobs[layer].data[0],None