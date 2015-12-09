# --------------------------------------------------------
# CCNN 
# Copyright (c) 2015 [See LICENSE file for details]
# Written by Deepak Pathak, Philipp Krahenbuhl
# --------------------------------------------------------

'''
To generate compact lmdb using convert_imageset tool (see /caffe-pathak/fcn_mil/src/create_imagenet.sh) :
-	You run the tool twice. Once to create the image lmdb and once to create the label lmdb. 
	The filename is the path to the image and label is always 0. Use --encode flag in both runs. 
	Use --grey flag for label.
- 	Encode flag is not present in python
-   See the shuffling code in generate_bb_indicator.m
'''

from __future__ import division
from config import *
import lmdb
import numpy as np
import scipy.stats, scipy.io
from PIL import Image

dataset = 'val'       # train or val or trainval
dirAddress = '/mnt/a/pathak/fcn_mil_cache/VOC2012'

inputs = np.loadtxt('{}/{}.txt'.format(dirAddress,dataset), str)

# Generate Image LMDB =====================================
image_db = lmdb.open('{}/images_{}_lmdb'.format(dirAddress,dataset), map_size=int(1e12))
with image_db.begin(write=True) as in_txn:
	for in_idx, in_ in enumerate(inputs):

		im = np.array(Image.open(dirAddress + '/JPEGImages/' + in_ + '.jpg'))  # numpy ndarray
		# Classes present : np.unique(im.astype(np.uint8))
        
		# If rgb image : im = im[:,:,::-1] (RGB to BGR); im = im.transpose((2, 0, 1)) (in caffe channel-height-width)
		# If ground truth single channel image : im = im.astype(np.uint8) and im = im[np.newaxis, :, :]
		im = im[:,:,::-1]
		im = im.transpose((2, 0, 1))
		im_dat = caffe.io.array_to_datum(im)
        
		# Note that the indices are zero padded to preserve their order: LMDB sorts the keys lexicographically so bare integers as strings will be disordered.
		in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
image_db.close()

# Generate GT LMDB =======================================
image_db = lmdb.open('{}/segmentation_class_{}_lmdb'.format(dirAddress,dataset), map_size=int(1e12))
with image_db.begin(write=True) as in_txn:
	for in_idx, in_ in enumerate(inputs):
		
		im = np.array(Image.open(dirAddress + '/SegmentationClassPNG/' + in_ + '.png'))  # numpy ndarray
		# Classes present : np.unique(im.astype(np.uint8))

		# If rgb image : im = im[:,:,::-1] (RGB to BGR); im = im.transpose((2, 0, 1)) (in caffe channel-height-width)
		# If ground truth single channel image : im = im.astype(np.uint8) and im = im[np.newaxis, :, :]
		im = im.astype(np.uint8)
		im = im[np.newaxis, :, :]
		im_dat = caffe.io.array_to_datum(im)

		# Note that the indices are zero padded to preserve their order: LMDB sorts the keys lexicographically so bare integers as strings will be disordered.
		in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
image_db.close()
