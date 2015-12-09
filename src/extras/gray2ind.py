# --------------------------------------------------------
# CCNN 
# Copyright (c) 2015 [See LICENSE file for details]
# Written by Deepak Pathak, Philipp Krahenbuhl
# --------------------------------------------------------

'''
- It is to be run after png2gray.py
- Converts the gray scale segmentation ground truth image to hdf5 format. 
'''

from sys import argv
from scipy.io import loadmat
import numpy as np
from PIL import Image,ImagePalette
import h5py
import random
random.seed(222)

# Code to convert image one by one ============

# if len(argv) < 3:
# 	print("Usage: %s png hf5"%argv[0])
# 	exit(1)

# N = 21
# im = Image.open(argv[1])
# I = np.array(im) 	# shape : (h,w)
# l = I[I>=0]			# shape : (hw,1)
# l = l[l<N]
# # im.close() : works with python3. No need to close in python2.7

# f = h5py.File(argv[2], "w")
# cnt = np.bincount(l,minlength=N)[None,:,None,None]
# f.create_dataset('cnt', cnt.shape, dtype='f')[...] = cnt
# f.create_dataset('indicator', cnt.shape, dtype='f')[...] = (cnt > 0).astype(float)
# f.create_dataset('indicator_0.01', cnt.shape, dtype='f')[...] = (cnt > 0.01*l.size).astype(float)
# f.create_dataset('indicator_0.05', cnt.shape, dtype='f')[...] = (cnt > 0.05*l.size).astype(float)
# f.create_dataset('indicator_0.10', cnt.shape, dtype='f')[...] = (cnt > 0.10*l.size).astype(float)
# f.close()

# =============================================
# =============================================


# Code to convert images in batch =============
# =============================================

# N = 21
# out_dir = 'SegmentationClassIndicator/'
# voc_dir = '/x/pathak/fcn_mil_cache/VOC2012/'
# imNames = open(voc_dir+out_dir+'val.txt','r')
# for line in imNames:
# 	im = Image.open(voc_dir+'SegmentationClassGray/'+line[:-1]+'.png')
# 	I = np.array(im) 	# shape : (h,w)
# 	l = I[I>=0]			# shape : (hw,1)
# 	l = l[l<N]
# 	# im.close() : works with python3. No need to close in python2.7

# 	f = h5py.File(voc_dir+out_dir+'/ClassIndicator/'+line[:-1]+'.hf5', "w")
# 	cnt = np.bincount(l,minlength=N)[None,:,None,None]
	
# 	#cnt = cnt[:,1:,:,:] # To ignore background class
	
# 	f.create_dataset('cnt', cnt.shape, dtype='f')[...] = cnt
# 	f.create_dataset('indicator', cnt.shape, dtype='f')[...] = (cnt > 0).astype(float)
# 	f.create_dataset('indicator_0.01', cnt.shape, dtype='f')[...] = (cnt > 0.01*l.size).astype(float)
# 	f.create_dataset('indicator_0.05', cnt.shape, dtype='f')[...] = (cnt > 0.05*l.size).astype(float)
# 	f.create_dataset('indicator_0.10', cnt.shape, dtype='f')[...] = (cnt > 0.10*l.size).astype(float)
# 	f.close()

# =============================================
# =============================================


# Code to generate annotations with the semi-supervised flags ====
# ================================================================

N = 21
out_dir = 'SegmentationClassIndicator/'
voc_dir = '/mnt/a/pathak/fcn_mil_cache/VOC2012/'
data = 'val'
imNames = open(voc_dir+data+'.txt','r')

classFreq = np.zeros(N)
for line in imNames:
	im = Image.open(voc_dir+'SegmentationClassGray/'+line[:-1]+'.png')
	I = np.array(im) 	# shape : (h,w)
	l = I[I>=0]			# shape : (hw,1)
	l = l[l<N]
	# im.close() : works with python3. No need to close in python2.7
	cnt = np.bincount(l,minlength=N)[None,:,None,None]
	classFreq += (cnt[0,:,0,0] > 0).astype(float)
#print classFreq.shape
print 'Class Frequency: ',classFreq
#classFreq = np.array([ 10578.,586. , 486. , 698. , 461. ,654.,385. ,1086.,1000.,1081. ,264.,528.,1177.,444.,482.,3898.,487.,299.,491.,500.,548.]) for train

classIm = []
for i in range(0,N):
	classIm.append([])
imID = 0
imNames = open(voc_dir+data+'.txt','r')
for line in imNames:
	im = Image.open(voc_dir+'SegmentationClassGray/'+line[:-1]+'.png')
	I = np.array(im)
	l = I[I>=0]
	l = l[l<N]
	cnt = np.bincount(l,minlength=N)[None,:,None,None]
	cnt = (cnt[0,:,0,0] > 0).astype(float)
	classesPresent = np.flatnonzero(cnt)
	classChosen = classesPresent[classFreq[classesPresent].argmin()]
	classIm[classChosen].append(imID)
	imID += 1

samples = [1,3,5,10,50,100,200]			# number of randomly sampled images per class
selectedImages = []
for i in range(0,len(samples)):
	selectedImages.append([])
	for j in range(1,N):
		temp = classIm[j]
		random.shuffle(temp)
		selectedImages[i].extend(temp[0:min(samples[i],len(temp))])
	selectedImages[i].sort()
print 'Images Selected'

imID = 0
temp = len(samples)*[0]
imNames = open(voc_dir+data+'.txt','r')
for line in imNames:
	im = Image.open(voc_dir+'SegmentationClassGray/'+line[:-1]+'.png')
	I = np.array(im) 	# shape : (h,w)
	l = I[I>=0]			# shape : (hw,1)
	l = l[l<N]
	# im.close() : works with python3. No need to close in python2.7

	f = h5py.File(voc_dir+out_dir+line[:-1]+'.hf5', "w")
	cnt = np.bincount(l,minlength=N)[None,:,None,None]
	
	f.create_dataset('cnt', cnt.shape, dtype='f')[...] = cnt
	f.create_dataset('indicator', cnt.shape, dtype='f')[...] = (cnt > 0).astype(float)
	f.create_dataset('indicator_0.01', cnt.shape, dtype='f')[...] = (cnt > 0.01*l.size).astype(float)
	f.create_dataset('indicator_0.05', cnt.shape, dtype='f')[...] = (cnt > 0.05*l.size).astype(float)
	f.create_dataset('indicator_0.10', cnt.shape, dtype='f')[...] = (cnt > 0.10*l.size).astype(float)
	
	for i in range(0,len(samples)):
		if temp[i]<len(selectedImages[i]) and imID == selectedImages[i][temp[i]]:
			f.create_dataset('flag_'+str(samples[i]), (1,), dtype='f')[...] = 1
			temp[i] += 1
		else:
			f.create_dataset('flag_'+str(samples[i]), (1,), dtype='f')[...] = 0

	f.close()
	imID += 1
print 'Samples: ',samples
print 'Temp: ',temp

# ===============================================
# ===============================================
