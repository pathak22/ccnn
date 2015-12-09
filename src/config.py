# --------------------------------------------------------
# CCNN 
# Copyright (c) 2015 [See LICENSE file for details]
# Written by Deepak Pathak, Philipp Krahenbuhl
# --------------------------------------------------------

from os import environ

def tryLoad(name, default):
    try:
        import user_config
    except:
        return None
    if hasattr(user_config, name):
        return getattr(user_config, name)
    return default

CAFFE_DIR = tryLoad('CAFFE_DIR', '.')

import sys
import config
PD = CAFFE_DIR + '/python'
if PD not in sys.path:
    sys.path.append(PD)

# if not 'GLOG_minloglevel' in environ:
environ['GLOG_minloglevel'] = '1'
# To supress the output level to command line you need to increase the loglevel to at least 2. Do it before importing caffe.
# 0 - debug
# 1 - info (still a LOT of outputs)
# 2 - warnings
# 3 - errors

import caffe
