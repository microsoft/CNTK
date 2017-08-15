# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from utils.nms.nms_wrapper import apply_nms_to_test_set_results

def evaluate_detections(all_boxes, all_gt_infos, classes, use_07_metric=False, apply_mms=True, nms_threshold=0.5, conf_threshold=0.0):
    '''
    Computes per-class average precision.

    Args:
        all_boxes:          shape of all_boxes: e.g. 21 classes x 4952 images x 58 rois x 5 coords+score
        all_gt_infos:       a dictionary that contains all ground truth annoations in the following form:
                            {'class_A': [{'bbox': array([[ 376.,  210.,  456.,  288.,   10.]], dtype=float32), 'det': [False], 'difficult': [False]}, ... ]}
                            'class_B': [ <bbox_list> ], <more_class_to_bbox_list_entries> }
        classes:            a list of class name, e.g. ['__background__', 'avocado', 'orange', 'butter']
        use_07_metric:      whether to use VOC07's 11 point AP computation (default False)
        apply_mms:          whether to apply non maximum suppression before computing average precision values
        nms_threshold:      the threshold for discarding overlapping ROIs in nms
        conf_threshold:     a minimum value for the score of an ROI. ROIs with lower score will be discarded

    Returns:
        aps - average precision value per class in a dictionary {classname: ap}
    '''

    if apply_mms:
        print ("Number of rois before non-maximum suppression: %d" % sum([len(all_boxes[i][j]) for i in range(len(all_boxes)) for j in range(len(all_boxes[0]))]))
        nms_dets,_ = apply_nms_to_test_set_results(all_boxes, nms_threshold, conf_threshold)
        print ("Number of rois  after non-maximum suppression: %d" % sum([len(nms_dets[i][j]) for i in range(len(all_boxes)) for j in range(len(all_boxes[0]))]))
    else:
        print ("Skipping non-maximum suppression")
        nms_dets = all_boxes

    aps = {}
    for classIndex, className in enumerate(classes):
        if className != '__background__':
            rec, prec, ap = _evaluate_detections(classIndex, nms_dets, all_gt_infos[className], use_07_metric=use_07_metric)
            aps[className] = ap

    return aps

def _evaluate_detections(classIndex, all_boxes, gtInfos, overlapThreshold=0.5, use_07_metric=False):
    '''
    Top level function that does the PASCAL VOC evaluation.
    '''

    # parse detections for this class
    # shape of all_boxes: e.g. 21 classes x 4952 images x 58 rois x 5 coords+score
    num_images = len(all_boxes[0])
    detBboxes = []
    detImgIndices = []
    detConfidences = []
    for imgIndex in range(num_images):
        dets = all_boxes[classIndex][imgIndex]
        if dets != []:
            for k in range(dets.shape[0]):
                detImgIndices.append(imgIndex)
                detConfidences.append(dets[k, -1])
                # the VOCdevkit expects 1-based indices
                detBboxes.append([dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1])
    detBboxes = np.array(detBboxes)
    detConfidences = np.array(detConfidences)

    # compute precision / recall / ap
    rec, prec, ap = _voc_computePrecisionRecallAp(
        class_recs=gtInfos,
        confidence=detConfidences,
        image_ids=detImgIndices,
        BB=detBboxes,
        ovthresh=overlapThreshold,
        use_07_metric=use_07_metric)
    return rec, prec, ap

def computeAveragePrecision(recalls, precisions, use_07_metric=False):
    '''
    Computes VOC AP given precision and recall.
    '''
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrecalls = np.concatenate(([0.], recalls, [1.]))
        mprecisions = np.concatenate(([0.], precisions, [0.]))

        # compute the precision envelope
        for i in range(mprecisions.size - 1, 0, -1):
            mprecisions[i - 1] = np.maximum(mprecisions[i - 1], mprecisions[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrecalls[1:] != mrecalls[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrecalls[i + 1] - mrecalls[i]) * mprecisions[i + 1])
    return ap

def _voc_computePrecisionRecallAp(class_recs, confidence, image_ids, BB, ovthresh=0.5, use_07_metric=False):
    '''
    Computes precision, recall. and average precision
    '''
    if len(BB) == 0:
        return 0.0, 0.0, 0.0

    # sort by confidence
    sorted_ind = np.argsort(-confidence)

    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    npos = sum([len(cr['bbox']) for cr in class_recs])
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = computeAveragePrecision(rec, prec, use_07_metric)
    return rec, prec, ap
