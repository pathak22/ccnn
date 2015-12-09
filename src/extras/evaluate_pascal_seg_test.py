# --------------------------------------------------------
# CCNN 
# Copyright (c) 2015 [See LICENSE file for details]
# Written by Deepak Pathak, Philipp Krahenbuhl
# --------------------------------------------------------

from __future__ import division
from pylab import *
from config import *
import os
from PIL import Image

# For ccnn fcn32 code trained model : size
net_def = '../../models/fcn_32s/train_32s.prototxt'
weights = '../../models/ccnn_models/ccnn_tag_size_trainval.caffemodel'
save_dir = '/mnt/a/pathak/fcn_mil_cache/visualized_output/seg12test_size_untuned/results/VOC2012/Segmentation/comp6_test_cls/'

caffe.set_device(2)
caffe.set_mode_gpu()
test_net = caffe.Net(net_def, weights, caffe.TEST)

fnames_test = np.loadtxt('/mnt/a/pathak/fcn_mil_cache/VOC2012/test.txt', str)

#os.makedirs(save_dir)
count = 0
for fname in fnames_test:
    count = count + 1
    if count % 10 == 1:
        print count
    # print fname
    test_net.forward()
    im = Image.fromarray(test_net.blobs['upscore-crf'].data[0,0].astype(np.uint8), mode='P')
    im.save(os.path.join(save_dir, fname + '.png'))
print 'Total Images : ',count
print 'Weight File : ', weights