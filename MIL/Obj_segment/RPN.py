from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import logging
import os
import sys
import yaml

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import load_cfg
from detectron.core.config import merge_cfg_from_cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
import detectron.core.rpn_generator as rpn_engine
import detectron.core.test_engine as model_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)



def get_rpn_box_proposals(im):
    cfg.immutable(False)
    merge_cfg_from_file('Obj_segment/rpn_R-50-C4_1x.yaml')
    cfg.NUM_GPUS = 1
    cfg.MODEL.RPN_ONLY = True
    cfg.TEST.RPN_PRE_NMS_TOP_N = 100
    cfg.TEST.RPN_POST_NMS_TOP_N = 10
    assert_and_infer_cfg(cache_urls=False)

    model = model_engine.initialize_model_from_cfg('Obj_segment/model_finalv5.pkl')
    with c2_utils.NamedCudaScope(0):
        boxes, scores = rpn_engine.im_proposals(model, im)
    return boxes, scores


def main():
    im = cv2.imread('/home/pazagra/T2.jpg')
    proposal_boxes, _proposal_scores = get_rpn_box_proposals(im)
    for bbox in proposal_boxes[:,]:
        vis_utils.vis_bbox(im,bbox)
    cv2.imshow("DD",im)
    cv2.waitKey()



if __name__ == '__main__':
    os.environ["GLOG_minloglevel"] = "3"
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=3'])
    setup_logging(__name__)
    main()