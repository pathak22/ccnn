# --------------------------------------------------------
# CCNN 
# Copyright (c) 2015 [See LICENSE file for details]
# Written by Deepak Pathak, Philipp Krahenbuhl
# --------------------------------------------------------

'''
- It is to be run after mat2png.py
- Converts png segmentation ground truth images to grayscale representing the labels as intensities.
- Use python3 to run.
'''

from sys import argv
from scipy.io import loadmat
import numpy as np
from PIL import Image,ImagePalette

if len(argv) < 3:
	print("Usage: %s png png"%argv[0])
	exit(1)

im = Image.open(argv[1])
Image.frombytes('L',im.size,im.tobytes()).save(argv[2])
