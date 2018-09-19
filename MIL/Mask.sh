#!/bin/env bash
source /home/pazagra/tfp3/bin/activate
export GLOG_minloglevel=2
python3 /home/pazagra/tfp3/mask/Mask_RCNN/Get_P.py
deactivate
