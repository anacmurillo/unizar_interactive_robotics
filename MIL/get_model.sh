wget -nc --directory-prefix=./ 		http://posefs1.perception.cs.cmu.edu/Users/ZheCao/pose_iter_440000.caffemodel
mv pose_iter_440000.caffemodel Interaction/model/pose_iter_440000.caffemodel
wget -nc --directory-prefix=./ http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
mv bvlc_reference_caffenet.caffemodel Descriptors/models/bvlc_reference_caffenet.caffemodel
