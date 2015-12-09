# --------------------------------------------------------
# CCNN 
# Copyright (c) 2015 [See LICENSE file for details]
# Modified by Deepak Pathak
# Originally written by Jonathan Long
# --------------------------------------------------------

from __future__ import division
import numpy as np
import os
from PIL import Image
from datetime import datetime
from config import *

fnames_val = {'pascal': np.loadtxt('/mnt/a/pathak/fcn_mil_cache/VOC2012/val.txt', str)}

def prepare():
    save_dir = '/mnt/a/pathak/fcn_mil_cache/visualized_output/2012val_best_raw/'
    return save_dir

def compute_hist(net, save_dir, dataset):
    n_cl = net.blobs['score_crop'].channels
    count = 1
    hist = np.zeros((n_cl, n_cl))
    for fname in fnames_val[dataset]:        
        net.forward()
        h, _, _ = np.histogram2d(net.blobs['gt'].data[0, 0].flatten(),
                net.blobs['score_crop'].data[0].argmax(0).flatten(),
                bins=n_cl, range=[[0, n_cl], [0, n_cl]])
        hist += h
        iu = np.zeros(n_cl)
        for i in range(n_cl):
            iu[i] = hist[i, i] / (hist[i].sum() + hist[:, i].sum() - hist[i, i])
        print 'Image : ',count,' ,  Name : ',fname,' , mean IU (till here) : ', np.nanmean(iu)*100
        #print '\tClasses Present : ',np.unique(net.blobs['gt'].data[0, 0].astype(np.uint8))
        #print '\tClasses Predicted : ', np.unique(net.blobs['score_crop'].data[0].argmax(0).astype(np.uint8))
        #print ''
        # im = Image.fromarray(net.blobs['score_crop'].data[0].argmax(0).astype(np.uint8), mode='P')
        # im.save(os.path.join(save_dir, fname + '.png'))
        count += 1
        import sys
        sys.stdout.flush()
    return hist

def seg_tests(test_net, save_format, dataset, weights, net_def):
    print '>>>', datetime.now(), 'Begin seg tests'
    n_cl = test_net.blobs['score_crop'].channels
    hist = compute_hist(test_net, save_format, dataset)
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'overall accuracy', acc
    # per-class accuracy
    acc = np.zeros(n_cl)
    for i in range(n_cl):
        acc[i] = hist[i, i] / hist[i].sum()
    print '>>>', datetime.now(), 'mean accuracy', np.nanmean(acc)
    # per-class IU
    iu = np.zeros(n_cl)
    for i in range(n_cl):
        iu[i] = hist[i, i] / (hist[i].sum() + hist[:, i].sum() - hist[i, i])
    print '>>>', datetime.now(), 'mean IU', np.nanmean(iu)*100
    iu2 = [ round(100*elem, 1) for elem in iu ]
    print '>>>', datetime.now(), 'per-class IU', iu2
    freq = hist.sum(1) / hist.sum()
    print '>>>', datetime.now(), 'fwavacc', \
            (freq[freq > 0] * iu[freq > 0]).sum()
    print 'Weight File', weights
    print 'Proto File', net_def


# Running the code 

dataset = 'pascal'
save_format = prepare()

net_def = '../models/fcn_32s/train_32s.prototxt'
weights = '../models/ccnn_models/ccnn_tag_size_train.caffemodel'

caffe.set_device(0)
caffe.set_mode_gpu()
test_net = caffe.Net(net_def, weights, caffe.TEST)
seg_tests(test_net, save_format, dataset, weights, net_def)
