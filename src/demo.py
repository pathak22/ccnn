# --------------------------------------------------------
# CCNN 
# Copyright (c) 2015 [See LICENSE file for details]
# Written by Deepak Pathak, Philipp Krahenbuhl
# --------------------------------------------------------

from config import *
import numpy as np
from PIL import Image

# Function to get VOC color map. Don't change the image being loaded here.
def palette_gt(gt):
    palette_im = Image.open('../models/examples/gt1.png')
    gt.putpalette(palette_im.palette)
    return gt

# Network definitions
net_def = '../models/fcn_32s/deploy_32s.prototxt'
weights = '../models/ccnn_models/ccnn_tag_size_train.caffemodel'

# Load Network
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(net_def, weights, caffe.TEST)

# Load Image
im = Image.open('../models/examples/im2.jpg')
im = np.array(im, dtype=np.float32)
im = im[:,:,::-1]               # Change to BGR
mean = np.array((104.00698793,116.66876762,122.67891434))
im -= mean     # Mean Subtraction
im = im.transpose(2,0,1)        # Blob: C x H x W
im = im[None,:,:,:]

# Assign Data
net.blobs['data'].reshape(*im.shape)
net.blobs['data'].data[...] = im
net.blobs['data-orig'].reshape(*im.shape)
net.blobs['data-orig'].data[...] = im+mean[None,:,None,None]

# Run forward
net.forward()
out = Image.fromarray(net.blobs['upscore-crf'].data[0,0].astype(np.uint8), mode='P')
out = palette_gt(out)
out.save('../models/examples/result.png')

# Classes Predicted
print 'Classes Predicted:', np.unique(net.blobs['upscore-crf'].data[0,0].astype(np.uint8))
print 'Result saved'
