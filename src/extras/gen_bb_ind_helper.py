# --------------------------------------------------------
# CCNN 
# Copyright (c) 2015 [See LICENSE file for details]
# Written by Deepak Pathak, Philipp Krahenbuhl
# --------------------------------------------------------


'''
- Converts the matlab generated indicator file to hdf5 format. It is used after generate_bb_indicator.m
'''

from sys import argv
from scipy.io import loadmat
import numpy as np
import h5py
import os

out_dir = 'trainList_cl12_seg12'
voc_dir = '/mnt/a/pathak/fcn_mil_cache/VOC2012/'
indicatorLabels = np.loadtxt(voc_dir+out_dir+'/train_labels.txt')

if not os.path.exists(voc_dir+out_dir+'/ClassIndicator'):
	print 'Creating Directory : '+voc_dir+out_dir+'/ClassIndicator';
	os.makedirs(voc_dir+out_dir+'/ClassIndicator')


imNames = open(voc_dir+out_dir+'/train.txt','r')
i = 0
for line in imNames:
	label = indicatorLabels[i,:]
	label = label[None,:,None,None]

	f = h5py.File(voc_dir+out_dir+'/ClassIndicator/'+line[:-1]+'.hf5', "w")
	f.create_dataset('indicator', label.shape, dtype='f')[...] = label
	f.close()

	i = i+1

imNames.close()
