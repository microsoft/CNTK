# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os, sys, argparse, copy
import cntk as C
from cntk import ops

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "..", "Image", "Detection", "FasterRCNN"))

C.device.try_set_default_device(C.device.cpu())

from FasterRCNN import eval_faster_rcnn_mAP, set_global_vars
from config import cfg

ops.register_native_user_function('ProposalLayerOp', 'Cntk.ProposalLayerLib-' + C.__version__.rstrip('+'), 'CreateProposalLayer')

def clone_with_native_proposal_layer(model):
    def filter(x): 
        return type(x) == C.Function and \
            x.op_name == 'UserFunction' and \
            x.name == 'ProposalLayer'

    def converter(x):
        layer_config = copy.deepcopy(x.attributes)
        layer_config["test_pre_nms_topN"] = cfg["TEST"].RPN_PRE_NMS_TOP_N
        layer_config["test_post_nms_topN"] = cfg["TEST"].RPN_POST_NMS_TOP_N
        layer_config["test_nms_thresh"] = float(cfg["TEST"].RPN_NMS_THRESH)
        layer_config["test_min_size"] = float(cfg["TEST"].RPN_MIN_SIZE)

        layer_config["train_pre_nms_topN"] = cfg["TRAIN"].RPN_PRE_NMS_TOP_N
        layer_config["train_post_nms_topN"] = cfg["TRAIN"].RPN_POST_NMS_TOP_N
        layer_config["train_nms_thresh"] = float(cfg["TRAIN"].RPN_NMS_THRESH)
        layer_config["train_min_size"] = float(cfg["TRAIN"].RPN_MIN_SIZE)
        
        return ops.native_user_function('ProposalLayerOp', list(x.inputs), layer_config, 'native_proposal_layer')

    return C.misc.convert(model, filter, converter)

def convert(model_path):
    device = C.cpu()
    model = C.Function.load(model_path, device=device)

    # Replace all python proposal layer user-functions with native proposal layer
    # user functions.
    return clone_with_native_proposal_layer(model)

def evaluate(model_path):
    # ProposalLayer currently only runs on the CPU
    eval_device = C.cpu()
    model = C.Function.load(model_path, device=eval_device)
    set_global_vars(False)
    return eval_faster_rcnn_mAP(model)

############################# 
# main function boilerplate #
#############################

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', '--model_path', 
        help='Filepath of a model created with FasterRCNN.py', required=True)
    parser.add_argument('-eval', '--eval_model', 
        help='Evaluate a FasterRCNN model (with or without a native Proposal Layer)', 
        required=False, default=False)
    
    args = parser.parse_args()

    if args.eval_model:
        evaluate(args.model_path)
    else:
        model = convert(args.model_path)
        path = os.path.dirname(args.model_path)
        filename = 'native_proposal_layer_' +  os.path.basename(args.model_path)
        model.save(os.path.join(path, filename))

    
    
    
    
  


    

