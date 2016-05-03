## CCNN: Constrained Convolutional Neural Networks for Weakly Supervised Segmentation

[Deepak Pathak](http://cs.berkeley.edu/~pathak), [Philipp Kr&auml;henb&uuml;hl](http://www.philkr.net/), [Trevor Darrell](http://cs.berkeley.edu/~trevor)

**CCNN** is a framework for optimizing convolutional neural networks with linear constraints.
 - It has been shown to achieve state-of-the-art results on the task of weakly-supervised semantic segmentation.
 - It is written in Python and C++, and based on [Caffe](http://caffe.berkeleyvision.org/).
 - It has been published at **ICCV 2015**. It was initially described in the [arXiv report](http://arxiv.org/abs/1506.03648).

If you find CCNN useful in your research, please cite:

    @inproceedings{pathakICCV15ccnn,
        Author = {Pathak, Deepak and Kr\"ahenb\"uhl, Philipp and Darrell, Trevor},
        Title = {Constrained Convolutional Neural Networks for Weakly Supervised Segmentation},
        Booktitle = {International Conference on Computer Vision ({ICCV})},
        Year = {2015}
    }

### License

CCNN is released under academic, non-commercial UC Berkeley license (see [LICENSE](https://github.com/pathak22/ccnn/blob/master/LICENSE) file for details). 

### Contents
1. [Requirements](#1-requirements)
2. [Installation](#2-installation)
3. [Usage](#3-usage)
4. [Scripts Information](#4-scripts-information)
5. [Extra Downloads](#5-extra-downloads)

### 1) Requirements

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))
2. GCC version more than 4.7
3. Boost version more than 1.53 (recommended). If system dependencies give issues, install anaconda dependencies:

  ```
  $ conda install boost
  $ conda install protobuf
  ```
  
4. A good GPU (e.g., Titan, K20, K40, ...) with at least 3G of memory is sufficient.

### 2) Installation

1. Clone the CCNN repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/pathak22/ccnn.git
  ```
  
2. Build Caffe and pycaffe

  - Now follow the Caffe installation instructions [here](http://caffe.berkeleyvision.org/installation.html)
  - Caffe *must* be built with support for Python layers!
  - In your Makefile.config, make sure to have this line uncommented
    `WITH_PYTHON_LAYER := 1`
  - You can download my [Makefile.config](http://www.cs.berkeley.edu/~pathak/ccnn/Makefile.config) for reference.
  
  ```Shell
  cd ccnn/caffe-ccnn
  # If you have all caffe requirements installed
  # and your Makefile.config in place, then simply do:
  make -j8 && make pycaffe
  ```
    
3. Now build CCNN

    ```Shell
    cd ccnn
    mkdir build
    cd build
    cmake ..
    make -j8
    ```
    
  - **Note:** If anaconda is installed, then python paths may have been messed b/w anaconda and system python. 
  - I usually run this command : 

  ```Shell
  cmake .. -DBOOST_ROOT=/home/pathak/anaconda -DPYTHON_LIBRARY=/home/pathak/anaconda/lib/libpython2.7.so -DPYTHON_INCLUDE_DIR=/home/pathak/anaconda/include/python2.7/ -DCMAKE_C_COMPILER=gcc-4.8 -DCMAKE_CXX_COMPILER=g++-4.8
  ```
  
  - To verify this do : `ccmake ./` inside the build folder and manually check the following things : 
  `MAKE_CXX_COMPILER, CMAKE_C_COMPILER , PYTHON_EXECUTABLE , PYTHON_INCLUDE_DIR , PYTHON_LIBRARY`
  - Make sure that cmake doesn't mess the anaconda boost to system boost.

4. Configure path (if needed) in `src/user_config.py`.

5. (Optional -- I don't do it) If everything runs fine, set `CMAKE_BUILD_TYPE` using `ccmake .` to `Release`. This prevents eigen from checking all assertions etc. and works faster.

### 3) Usage

**Demo** CCNN.

```Shell
cd ccnn
bash ./models/scripts/download_ccnn_models.sh
# This will populate the `ccnn/models/` folder with trained models.
python ./src/demo.py
```

**Train** CCNN.

```Shell
cd ccnn
bash ./models/scripts/download_pretrained_models.sh
# This will populate the `ccnn/models/` folder with imagenet pre-trained models.
python ./src/train.py 2> log.txt
```

**Test** CCNN.

```Shell
cd ccnn
python ./src/test.py  # To test IOU with CRF post-processing
python ./src/test_argmax.py  # To test IOU without CRF
```

### 4) Scripts Information

Model Prototxts:
- `models/fcn_8s/` : Atrous algorithm based 8-strided VGG, described [here](http://arxiv.org/abs/1412.7062).
- `models/fcn_32s/` : 32-strided VGG

Configure:
- `src/config.py` : Set glog-minlevel accordingly to get desired caffe output to terminal

Helper Scripts:
- `src/extras/` : These scripts are not needed to run the code. They are simple helper scripts to create data, to prepare pascal test server file, to add pascal cmap to segmentation outputs etc.

### 5) Extra Downloads

- Pascal VOC Image List: [train](http://www.cs.berkeley.edu/~pathak/ccnn/train.txt), [val](http://www.cs.berkeley.edu/~pathak/ccnn/val.txt), [trainval](http://www.cs.berkeley.edu/~pathak/ccnn/trainval.txt), [test](http://www.cs.berkeley.edu/~pathak/ccnn/test.txt)
- [Training image-level label indicator files](http://www.cs.berkeley.edu/~pathak/ccnn/trainIndicatorFiles.tar.gz)
- [Pascal VOC 2012 validation result images](http://www.cs.berkeley.edu/~pathak/ccnn/voc_2012_val_results.tar.gz)
- [Pascal VOC 2012 test result images](http://www.cs.berkeley.edu/~pathak/ccnn/voc_2012_test_results.tar.gz)
