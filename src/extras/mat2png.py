# --------------------------------------------------------
# CCNN 
# Copyright (c) 2015 [See LICENSE file for details]
# Written by Deepak Pathak, Philipp Krahenbuhl
# --------------------------------------------------------

'''
- Converts the mat file provided by Hariharan et. al. SBD to png single channel images with color palette.
- Use python3 to run. After this, use png2gray.py and then gray2ind.py.
'''

from sys import argv
from scipy.io import loadmat
import numpy as np
from PIL import Image

if len(argv) < 3:
	print("Usage: %s mat png"%argv[0])
	exit(1)

M = loadmat( argv[1] )
im = Image.fromarray(M['GTcls'][0,0]['Segmentation'].astype(np.uint8),'P')

if len(argv)>3:
	im.putpalette( Image.open( argv[3] ).palette )

im.save(argv[2])
