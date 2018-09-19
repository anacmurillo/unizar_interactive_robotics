from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import numpy as np
from collections import defaultdict
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import time
from Obj_segment.Rect import *
from caffe2.python import workspace
from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)
class maskrcnn:
    def __init__(self,kp=False):
        workspace.SwitchWorkspace("caffe2", True)
        workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
        if kp:
            merge_cfg_from_file("Obj_segment/e2e_keypoint_rcnn_X-101-32x8d-FPN_s1x.yaml")
        else:
            # merge_cfg_from_file("Obj_segment/e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml")
            # merge_cfg_from_file("Obj_segment/e2e_mask_rcnn_R-101-FPN_2x.yaml")
            # merge_cfg_from_file("Obj_segment/e2e_mask_rcnn_R-50-C4_2x.yaml")
            merge_cfg_from_file("Obj_segment/rpn_R-50-FPN_1x.yaml")
        cfg.immutable(False)
        cfg.NUM_GPUS = 1
        assert_and_infer_cfg(cache_urls=False)
        if kp:
            self.model = infer_engine.initialize_model_from_cfg("Obj_segment/model_final_kp.pkl")
        else:
            self.model = infer_engine.initialize_model_from_cfg("Obj_segment/model_finalv4.pkl")
            # self.model = infer_engine.initialize_model_from_cfg("Obj_segment/model_final.pkl")

    def maskrcnn(self,im):
        img = cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        workspace.SwitchWorkspace("caffe2",True)
        listado = []
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(self.model, img, None, )
            print "EEEEEEEEEEEE"
            # im = vis_utils.vis_one_image_opencv(im,cls_boxes,cls_segms,cls_keyps,show_box=True,thresh=0.0)
            box,seg,kp,clas = vis_utils.convert_from_cls_format(cls_boxes, cls_segms, cls_keyps)
            i = 0
            for b in box:
                (x0, y0, w, h) = (b[0], b[1], b[2] - b[0], b[3] - b[1])
                x1, y1 = int(x0 + w), int(y0 + h)
                x0, y0 = int(x0), int(y0)
                listado.append((x0,y0,int(w),int(h),clas[i]))
                i+=1
        return listado,kp
