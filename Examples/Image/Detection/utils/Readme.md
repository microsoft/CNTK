## Detection utils

This folder contains Python modules that are utilities for object detection networks. 

### Cython modules

To use the rpn component you need precompiled cython modules for nms (at least cpu_nms.cpXX-win_amd64.pyd for Windows or cpu_nms.cpython-XXm.so for Linux) and  bbox (cython_bbox.cpXX-win_amd64.pyd for Windows or cython_bbox.cpython-XXm.so for Linux). 
To compile the cython modules for windows see (https://github.com/MrGF/py-faster-rcnn-windows): 
```
git clone https://github.com/MrGF/py-faster-rcnn-windows
cd $FRCN_ROOT/lib
python setup.py build_ext --inplace
```
For Linux see (https://github.com/rbgirshick/py-faster-rcnn):
```
git clone https://github.com/rbgirshick/py-faster-rcnn
cd $FRCN_ROOT/lib
python setup.py build_ext --inplace
```
Copy the compiled `.pyd` (Windows) or `.so` (Linux) files into the `cython_modules` subfolder of this utils folder.

##### `default_config`

Contains all required parameters for using a region proposal network in training or evaluation. You can overwrite these parameters by specifying a `config.py` file of the same format inside your working directory.

### `rpn` module overview

The rpn module contains helper methods and required layers to generate region proposal networks for object detection.

##### `rpn_helpers`

Contains helper methods to create a region proposal network (rpn) and a proposal target layer for training the rpn.

##### `generate_anchors.py`

Generates a regular grid of multi-scale, multi-aspect anchor boxes.

##### `proposal_layer.py`

Converts RPN outputs (per-anchor scores and bbox regression estimates) into object proposals.

##### `anchor_target_layer.py` 

Generates training targets/labels for each anchor. Classification labels are 1 (object), 0 (not object) or -1 (ignore).
Bbox regression targets are specified when the classification label is > 0.

##### `proposal_target_layer.py`

Generates training targets/labels for each object proposal: classification labels 0 - K (bg or object class 1, ... , K)
and bbox regression targets in that case that the label is > 0.

##### `generate.py`

Generate object detection proposals from an imdb using an RPN.
