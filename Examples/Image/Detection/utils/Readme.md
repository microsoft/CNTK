# Object detection utils

## Region proposal networks

The folder 'rpn' contains CNTK user functions that provide the necessary layers to build a region proposal network for object detection as proposed in the "Faster R-CNN" paper:

    Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun:
    "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"

The file `rpn_helpers.py` in that folder provides the following two helper methods to get started:
 - create_rpn: creates a region proposal network from the provided conv feature map and the ground truth boxes. It return both the proposed regions and the rpn losses.
 - create_proposal_target_layer: creates a proposal target layer for training an object detection network. It returns rois along with the target labels and target bbox regression coefficents for training.
 
 