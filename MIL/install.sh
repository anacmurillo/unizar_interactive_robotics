sudo apt-get install python3-pip python3-dev python-virtualenv 
virtualenv --system-site-packages -p python3 tfenv
source tfenv/bin/activate
easy_install -U pip
pip3 install --upgrade tensorflow
pip3 install scikit-image
pip3 install cython
pip3 install keras
pip3 install h5py
wget -nc --directory-prefix=./ https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
mv mask_rcnn_coco.h5 tfenv/maskrcnn/Mask_RCNN/mask_rcnn_coco.h5
cd tfenv/maskrcnn/cocotools/coco/PythonAPI
make install
