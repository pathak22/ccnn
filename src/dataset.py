# --------------------------------------------------------
# CCNN 
# Copyright (c) 2015 [See LICENSE file for details]
# Written by Deepak Pathak, Philipp Krahenbuhl
# --------------------------------------------------------

import numpy as np

VOC_DIR = '/mnt/a/pathak/fcn_mil_cache/VOC2012'
CHANNEL_MEAN = np.array([104.00698793,116.66876762,122.67891434])

def idsVOC(type='train'):
	if type == 'train':
		return [l.strip() for l in open(VOC_DIR+'/train.txt','r')]
	if type == 'trainval':
		return [l.strip() for l in open(VOC_DIR+'/trainval.txt','r')]
	if type == 'trainval':
		return [l.strip() for l in open(VOC_DIR+'/test.txt','r')]
	return [l.strip() for l in open(VOC_DIR+'/val.txt','r')]

t0,t1 = 0,0
def fetchVOC( id ):
	from skimage import io
	from time import time
	global t0,t1
	t0 += time()
	im = io.imread(VOC_DIR+"/JPEGImages/%s.jpg"%id)
	tim = im[:,:,::-1].transpose((2,0,1))-CHANNEL_MEAN[:,None,None]
	lbl = io.imread(VOC_DIR+"/SegmentationClassGray/%s.png"%id)
	t1 += time()
	return tim[None],lbl[None,None]

