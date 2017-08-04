# CNTK Examples: Image/Detection/Faster R-CNN

## Overview

This folder contains an end-to-end solution for using Faster R-CNN to perform object detection. 
The original research paper for Faster R-CNN can be found at [https://arxiv.org/abs/1506.01497](https://arxiv.org/abs/1506.01497).
Base models that are supported by the current configuration are AlexNet and VGG16. 
Two image set that are preconfigured are Pascal VOC 2007 and Grocery. 
Other base models or image sets can be used by adapting config.py.

## Running the example

### Setup

To run Faster R-CNN you need a CNTK Python environment. Install the following additional packages:

```
pip install opencv-python
pip install easydict
pip install pyyaml
pip install future
```

The code uses prebuild Cython modules for parts of the region proposal network (see `Examples/Image/Detection/utils/cython_modules`). 
These binaries are contained in the repository for Python 3.5 under Windows and Python 3.4 under Linux.
If you require other versions please follow the instructions at [https://github.com/rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#installation-sufficient-for-the-demo).

### Getting the data and AlexNet model

We use a toy dataset of images captured from a refrigerator to demonstrate Faster R-CNN. Both the dataset and the pre-trained AlexNet model can be downloaded by running the following Python command:

`python install_data_and_model.py`

After running the script, the toy dataset will be installed under the `Image/DataSets/Grocery` folder. And the AlexNet model will be downloaded to the `Image/PretrainedModels` folder. 
We recommend you to keep the downloaded data in the respective folder while downloading, as the configuration files in this folder assumes that by default.

### Running Faster R-CNN on the example data

To train and evaluate Faster R-CNN run 

`python FasterRCNN.py`

The results for end-to-end training on Grocery using AlexNet as the base model should look similar to these:

```
AP for          eggBox = 1.0000
AP for          tomato = 1.0000
AP for     orangeJuice = 1.0000
AP for         ketchup = 0.6667
AP for         mustard = 1.0000
AP for           water = 0.5000
AP for       champagne = 1.0000
AP for         joghurt = 1.0000
AP for          pepper = 1.0000
AP for         avocado = 1.0000
AP for           onion = 1.0000
AP for         tabasco = 1.0000
AP for            milk = 1.0000
AP for          orange = 1.0000
AP for          gerkin = 1.0000
AP for          butter = 1.0000
Mean AP = 0.9479
```

### Running Faster R-CNN on Pascal VOC data

To download the Pascal data and create the annotations file in CNTK format run the following scripts:

```
python Examples/Image/DataSets/Pascal/install_pascalvoc.py
python Examples/Image/DataSets/Pascal/mappings/create_mappings.py
```

Change the data set parameter in `config.py` to `'Pascal'`:

```
__C.CNTK.DATASET = "Pascal"    # "Grocery" or "Pascal"
```

Now you're set to train on the Pascal VOC 2007 data using `python FasterRCNN.py`. Beware that training might take a while.

### Running Faster R-CNN on your own data

Preparing your own data and annotating it with ground truth bounding boxes is describer [here](https://docs.microsoft.com/en-us/cognitive-toolkit/Object-Detection-using-Fast-R-CNN#train-on-your-own-data).
After storing your images in the described folder structure and annotating them please run

`python Examples/Image/Detection/utils/annotations/annotations_helper.py`

after changing the folder in that script to your data folder. Finally, in `config.py` you need to add a new section for your data set, e.g.:

```
if __C.CNTK.DATASET == "YourDataSet":
    __C.CNTK.MAP_FILE_PATH = "../../DataSets/YourDataSet"
    __C.CNTK.CLASS_MAP_FILE = "class_map.txt"
    __C.CNTK.TRAIN_MAP_FILE = "train_img_file.txt"
    __C.CNTK.TEST_MAP_FILE = "test_img_file.txt"
    __C.CNTK.TRAIN_ROI_FILE = "train_roi_file.txt"
    __C.CNTK.TEST_ROI_FILE = "test_roi_file.txt"
    __C.CNTK.NUM_TRAIN_IMAGES = 500
    __C.CNTK.NUM_TEST_IMAGES = 200
```

Set `__C.CNTK.DATASET = "YourDataSet"` and run `python FasterRCNN.py` to train and evaluate Faster R-CNN on your data.

## Technical details

### Parameters

All options and parameters are in `config.py`. These include

```
__C.CNTK.DATASET = "Grocery"    # "Grocery" or "Pascal"
__C.CNTK.BASE_MODEL = "AlexNet" # "VGG16" or "AlexNet"

__C.CNTK.TRAIN_E2E = True       # E2E or 4-stage training

__C.CNTK.E2E_MAX_EPOCHS = 20
__C.CNTK.E2E_LR_PER_SAMPLE = [0.001] * 10 + [0.0001] * 10 + [0.00001]
```

### Faster R-CNN CNTK code

Most of the code is in `FasterRCNN.py` (and `Examples/Image/Detection/utils/rpn/rpn_helpers.py` for the region proposal network). This is how the network is built in the CNTK Python API:

```
def create_faster_rcnn_predictor(base_model_file_name, features, scaled_gt_boxes, dims_input):
    # Load the pre-trained classification net and clone layers
    base_model = load_model(base_model_file_name)
    conv_layers = clone_conv_layers(base_model)
    fc_layers = clone_model(base_model, [pool_node_name], [last_hidden_node_name], clone_method=CloneMethod.clone)

    # Normalization and conv layers
    feat_norm = features - normalization_const
    conv_out = conv_layers(feat_norm)

    # RPN and prediction targets
    rpn_rois, rpn_losses = \
        create_rpn(conv_out, scaled_gt_boxes, dims_input, proposal_layer_param_string=cfg["CNTK"].PROPOSAL_LAYER_PARAMS)
    rois, label_targets, bbox_targets, bbox_inside_weights = \
        create_proposal_target_layer(rpn_rois, scaled_gt_boxes, num_classes=globalvars['num_classes'])

    # Fast RCNN and losses
    cls_score, bbox_pred = create_fast_rcnn_predictor(conv_out, rois, fc_layers)
    detection_losses = create_detection_losses(cls_score, label_targets, rois, bbox_pred, bbox_targets, bbox_inside_weights)
    loss = rpn_losses + detection_losses
    pred_error = classification_error(cls_score, label_targets, axis=1)

    return loss, pred_error

def create_fast_rcnn_predictor(conv_out, rois, fc_layers):
    # RCNN
    roi_out = roipooling(conv_out, rois, cntk.MAX_POOLING, (roi_dim, roi_dim), spatial_scale=1/16.0)
    fc_out = fc_layers(roi_out)

    # prediction head
    W_pred = parameter(shape=(4096, globalvars['num_classes']), init=normal(scale=0.01), name="cls_score.W")
    b_pred = parameter(shape=globalvars['num_classes'], init=0, name="cls_score.b")
    cls_score = plus(times(fc_out, W_pred), b_pred, name='cls_score')

    # regression head
    W_regr = parameter(shape=(4096, globalvars['num_classes']*4), init=normal(scale=0.001), name="bbox_regr.W")
    b_regr = parameter(shape=globalvars['num_classes']*4, init=0, name="bbox_regr.b")
    bbox_pred = plus(times(fc_out, W_regr), b_regr, name='bbox_regr')

    return cls_score, bbox_pred

def create_rpn(conv_out, scaled_gt_boxes, im_info, add_loss_functions=True,
               proposal_layer_param_string=None, conv_bias_init=0.0):
    rpn_conv_3x3 = Convolution((3, 3), num_channels, activation=relu, pad=True, strides=1,
                                init = normal(scale=0.01), init_bias=conv_bias_init)(conv_out)
    rpn_cls_score = Convolution((1, 1), 18, activation=None, name="rpn_cls_score",
                                init = normal(scale=0.01), init_bias=conv_bias_init)(rpn_conv_3x3)  # 2(bg/fg)  * 9(anchors)
    rpn_bbox_pred = Convolution((1, 1), 36, activation=None, name="rpn_bbox_pred",
                                init = normal(scale=0.01), init_bias=conv_bias_init)(rpn_conv_3x3)  # 4(coords) * 9(anchors)

    # apply softmax to get (bg, fg) probabilities and reshape predictions back to grid of (18, H, W)
    num_predictions = int(rpn_cls_score.shape[0] / 2)
    rpn_cls_score_rshp = reshape(rpn_cls_score, (2, num_predictions, rpn_cls_score.shape[1], rpn_cls_score.shape[2]), name="rpn_cls_score_rshp")
    p_rpn_cls_score_rshp = cntk.placeholder()
    rpn_cls_sm = softmax(p_rpn_cls_score_rshp, axis=0)
    rpn_cls_prob = cntk.as_block(rpn_cls_sm, [(p_rpn_cls_score_rshp, rpn_cls_score_rshp)], 'Softmax', 'rpn_cls_prob')
    rpn_cls_prob_reshape = reshape(rpn_cls_prob, rpn_cls_score.shape, name="rpn_cls_prob_reshape")

    # proposal layer
    rpn_rois_raw = user_function(ProposalLayer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, param_str=proposal_layer_param_string))
    rpn_rois = alias(rpn_rois_raw, name='rpn_rois')
```

### Algorithm 

All details can be found in the original research paper: [https://arxiv.org/abs/1506.01497](https://arxiv.org/abs/1506.01497).

