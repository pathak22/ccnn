# --------------------------------------------------------
# CCNN 
# Copyright (c) 2015 [See LICENSE file for details]
# Written by Deepak Pathak, Philipp Krahenbuhl
# --------------------------------------------------------

from config import *
import ccnn
import python_layers, dataset
from glob import glob
import numpy as np
from time import time
from sys import argv

caffe.set_mode_gpu()
caffe.set_device(0)

MODEL_PROTOTXT = '../models/fcn_8s/train_8s.prototxt'
MODEL_INIT = '../models/imagenet_pretrained_models/vgg_init_8s.caffemodel'

# MODEL_PROTOTXT = '../models/fcn_32s/train_32s.prototxt'
# MODEL_INIT = '../models/imagenet_pretrained_models/vgg_init_32s.caffemodel'

MODEL_SAVE = '../models/ccnn_models/ccnn.caffemodel'

if len(argv)>1:
	MODEL_SAVE = argv[1]
doTest = False

SOLVER_STR = """train_net: "{TRAIN_NET}"
base_lr: 1e-6
lr_policy: "step"
gamma: 0.1
stepsize: 40000
display: 20
max_iter: 35000
momentum: 0.99
weight_decay: 0.0000005
#average_loss: 1
"""

SOLVER_STR = SOLVER_STR.replace( "{TRAIN_NET}", MODEL_PROTOTXT )

t0 = time()
solver = caffe.get_solver_from_string(SOLVER_STR)
solver.net.copy_from(MODEL_INIT) # Note that this does not copy the interpolation params!
print "Load model %fs"%(time()-t0)

for it in range(35):
	t0 = time()
	solver.step(1000)
	t1 = time()
	print "%4d iterations t ="%((it+1)*1000), t1-t0
	solver.net.save(MODEL_SAVE)
	if (it+1)%5==0 and it>10:
		solver.net.save(MODEL_SAVE + '_'+str(it+1))
	if doTest and it>30:
		import subprocess
		try:
			sp.wait()
		except:
			pass
		sp = subprocess.Popen(['python', 'test_argmax.py'])

if doTest:
	try:
		sp.wait()
	except:
		pass
