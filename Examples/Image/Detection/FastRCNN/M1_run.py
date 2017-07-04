# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from A2_RunWithPyModel import create_mb_source, train_fast_rcnn, base_path
import os
from cntk import *

import hierarchical_classification_tool as HCT

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, ".."))
from utils.map.map_helpers import evaluate_detections

if __name__ == '__main__':
    os.chdir(base_path)
    model_path = os.path.join(abs_path, "Output", "frcn_py.model")

    # Train only is no model exists yet
    if os.path.exists(model_path):
        print("Loading existing model from %s" % model_path)
        trained_model = load_model(model_path)
    else:
        trained_model = train_fast_rcnn()
        trained_model.save(model_path)
        print("Stored trained model at %s" % model_path)

    # eval trained_model
    print("\n---Evaluation---")

    output_mapper = HCT.tree_map.get_output_mapper()
    known_classes = output_mapper.get_all_classes()

    all_boxes=[] # classes * images * rois * vector_length
    all_gt_infos=[]

    import ipdb; ipdb.set_trace()

    aps=evaluate_detections(all_boxes, all_gt_infos, known_classes)

    ap_list = []
    for class_name in aps:
        ap_list += [aps[class_name]]
        print('AP for {:>15} = {:.4f}'.format(class_name, aps[class_name]))
    print('Mean AP = {:.4f}'.format(np.nanmean(ap_list)))