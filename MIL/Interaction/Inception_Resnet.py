import sys
import cv2
import numpy as np
import caffe
import time

class Inception_Resnet():

    def __init__(self):
        model_file = 'Descriptors/models/inception-resnet-v2.caffemodel'
        deploy_prototxt = 'Descriptors/models/deploy_inception-resnet-v2.prototxt'
        self.net = caffe.Net(deploy_prototxt, model_file, caffe.TEST)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_raw_scale('data', 255.0)

    def ComputeProb(self,RGB,Depth):
        start = time.time()
        layer = 'pool_8x8_s1_drop'
        if layer not in self.net.blobs:
            return None,None

        self.net.blobs['data'].reshape(1, 3, 299, 299)
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', RGB)

        output = self.net.forward()
        print time.time()-start
        return self.net.blobs[layer].data[0][:,0,0]
