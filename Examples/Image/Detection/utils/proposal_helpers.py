# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os, sys
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, ".."))

import numpy as np
import cv2
from utils.rpn.bbox_transform import bbox_transform
from utils.cython_modules.cython_bbox import bbox_overlaps

random_seed = 23
global ss_lib_loaded, find_candidate_object_locations
ss_lib_loaded = False

def load_selective_search_lib():
    global find_candidate_object_locations
    from dlib import find_candidate_object_locations as algo
    find_candidate_object_locations = algo

    global ss_lib_loaded
    ss_lib_loaded = True

def compute_image_stats(img_width, img_height, pad_width, pad_height):
    do_scale_w = img_width > img_height
    target_w = pad_width
    target_h = pad_height

    if do_scale_w:
        scale_factor = float(pad_width) / float(img_width)
        target_h = int(np.round(img_height * scale_factor))
    else:
        scale_factor = float(pad_height) / float(img_height)
        target_w = int(np.round(img_width * scale_factor))

    top = int(max(0, np.round((pad_height - target_h) / 2)))
    left = int(max(0, np.round((pad_width - target_w) / 2)))
    bottom = pad_height - top - target_h
    right = pad_width - left - target_w
    return [target_w, target_h, img_width, img_height, top, bottom, left, right, scale_factor]

def filterRois(rects, img_w, img_h, roi_min_area, roi_max_area, roi_min_side, roi_max_side, roi_max_aspect_ratio):
    filteredRects = []
    filteredRectsSet = set()
    for rect in rects:
        if tuple(rect) in filteredRectsSet: # excluding rectangles with same co-ordinates
            continue

        x, y, x2, y2 = rect
        w = x2 - x
        h = y2 - y
        assert(w>=0 and h>=0)

        # apply filters
        if h == 0 or w == 0 or \
           x2 > img_w or y2 > img_h or \
           w < roi_min_side or h < roi_min_side or \
           w > roi_max_side or h > roi_max_side or \
           w * h < roi_min_area or w * h > roi_max_area or \
           w / h > roi_max_aspect_ratio or h / w > roi_max_aspect_ratio:
               continue
        filteredRects.append(rect)
        filteredRectsSet.add(tuple(rect))

    # could combine rectangles using non-maximum surpression or with similar co-ordinates
    # groupedRectangles, weights = cv2.groupRectangles(np.asanyarray(rectsInput, np.float).tolist(), 1, 0.3)
    # groupedRectangles = nms_python(np.asarray(rectsInput, np.float), 0.5)
    assert(len(filteredRects) > 0)
    return filteredRects

def compute_proposals(img, num_proposals, cfg):
    img_w = len(img[0])
    img_h = len(img)

    if cfg is None: cfg = {}
    roi_ss_kvals = (10, 500, 5)                                     if 'roi_ss_kvals' not in cfg else tuple(cfg['roi_ss_kvals'])
    roi_ss_mm_iterations = 30                                       if 'roi_ss_mm_iterations' not in cfg else cfg['roi_ss_mm_iterations']
    roi_ss_min_size = 9                                             if 'roi_ss_min_size' not in cfg else cfg['roi_ss_min_size']
    roi_ss_img_size = 200                                           if 'roi_ss_img_size' not in cfg else cfg['roi_ss_img_size']
    roi_min_side_rel = 0.04                                         if 'roi_min_side_rel' not in cfg else cfg['roi_min_side_rel']
    roi_max_side_rel = 0.4                                          if 'roi_max_side_rel' not in cfg else cfg['roi_max_side_rel']
    roi_min_area_rel = 2 * roi_min_side_rel * roi_min_side_rel      if 'roi_min_area_rel' not in cfg else cfg['roi_min_area_rel']
    roi_max_area_rel = 0.33 * roi_max_side_rel * roi_max_side_rel   if 'roi_max_area_rel' not in cfg else cfg['roi_max_area_rel']
    roi_max_aspect_ratio = 4.0                                      if 'roi_max_aspect_ratio' not in cfg else cfg['roi_max_aspect_ratio']
    roi_grid_aspect_ratios = [1.0, 2.0, 0.5]                        if 'roi_grid_aspect_ratios' not in cfg else cfg['roi_grid_aspect_ratios']
    debug_output = False if not ('CNTK' in cfg and 'DEBUG_OUTPUT' in cfg.CNTK) else cfg.CNTK.DEBUG_OUTPUT

    scale = 1.0 * roi_ss_img_size / max(img.shape[:2])
    img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    roi_min_side = roi_min_side_rel * roi_ss_img_size
    roi_max_side = roi_max_side_rel * roi_ss_img_size
    roi_min_area = roi_min_area_rel * roi_ss_img_size * roi_ss_img_size
    roi_max_area = roi_max_area_rel * roi_ss_img_size * roi_ss_img_size

    if not ss_lib_loaded: load_selective_search_lib()
    rects = []
    tmp = []
    find_candidate_object_locations(img, tmp, kvals=roi_ss_kvals, min_size=roi_ss_min_size, max_merging_iterations=roi_ss_mm_iterations)
    for k, d in enumerate(tmp):
        rects.append([d.left(), d.top(), d.right(), d.bottom()])
    filtered_rects = filterRois(rects, img_w, img_h, roi_min_area, roi_max_area, roi_min_side, roi_max_side, roi_max_aspect_ratio)
    scaled_rects = np.array(filtered_rects) * (1/scale)
    if debug_output:
        print("selective search rois before | after filtering: {} | {}. Requested: {}".format(len(rects), len(filtered_rects), num_proposals))

    num_rects = scaled_rects.shape[0]
    np.random.seed(random_seed)
    if num_rects < num_proposals:
        try:
            shuffle = not cfg.CNTK.FORCE_DETERMINISTIC
        except:
            shuffle = True

        roi_min_side = roi_min_side_rel * min(img_w, img_h)
        roi_max_side = roi_max_side_rel * max(img_w, img_h)
        grid_proposals = compute_grid_proposals(num_proposals - num_rects, img_w, img_h, roi_min_side, roi_max_side, roi_grid_aspect_ratios, shuffle)
        scaled_rects = np.vstack([scaled_rects, grid_proposals])
    elif num_rects > num_proposals:
        keep_inds = range(num_rects)
        keep_inds = np.random.choice(keep_inds, size=num_proposals, replace=False)
        scaled_rects = scaled_rects[keep_inds]

    return scaled_rects

def compute_grid_proposals(num_proposals, img_w, img_h, min_wh, max_wh, aspect_ratios = [1.0, 2.0, 0.5], shuffle=True):
    rects = []
    iter = 0
    while len(rects) < num_proposals:
        if iter == 0:
            new_ar = aspect_ratios
        else:
            new_ar = []
            for ar in aspect_ratios:
                new_ar.append(ar * (0.9 ** iter))
                new_ar.append(ar * (1.1 ** iter))

        new_rects = np.array(_compute_grid_proposals(img_w, img_h, min_wh, max_wh, new_ar))
        take = min(num_proposals - len(rects), len(new_rects))

        if shuffle and take < len(new_rects):
            keep_inds = range(len(new_rects))
            keep_inds = np.random.choice(keep_inds, size=take, replace=False)
            new_rects = new_rects[keep_inds]
        else:
            new_rects = new_rects[:take]

        rects.extend(new_rects)
        iter = iter + 1

    np_rects = np.array(rects)
    assert np_rects.shape[0] == num_proposals
    return np_rects

def _compute_grid_proposals(img_w, img_h, min_wh, max_wh, aspect_ratios):
    rects = []
    cell_w = max_wh
    while cell_w >= min_wh:
        step = cell_w / 2.0
        for aspect_ratio in aspect_ratios:
            w_start = 0
            while w_start < img_w:
                h_start = 0
                while h_start < img_h:
                    if aspect_ratio < 1:
                        w_end = w_start + cell_w
                        h_end = h_start + cell_w / aspect_ratio
                    else:
                        w_end = w_start + cell_w * aspect_ratio
                        h_end = h_start + cell_w
                    if w_end < img_w-1 and h_end < img_h-1:
                        rects.append([int(w_start), int(h_start), int(w_end), int(h_end)])
                    h_start += step
                w_start += step
        cell_w = cell_w / 2

    return rects

def write_to_file(proposal_list, filename):
    with open(filename, 'w') as f:
        for i in range(len(proposal_list)):
            proposals = proposal_list[i]
            line = "{}\t".format(i)
            for p in proposals:
                line = "{} {}".format(line, " ".join([str(v) for v in p]))
            f.write(line)

def compute_targets(proposals, gt_rois, iou_threshold, normalize_means, normalize_stds):
    """Compute bounding-box regression targets for an image."""
    if len(gt_rois) == 0:
        # Bail if the image has no ground-truth ROIs
        return np.zeros((proposals.shape[0], 6), dtype=np.float32)

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = bbox_overlaps(
        np.ascontiguousarray(proposals, dtype=np.float),
        np.ascontiguousarray(gt_rois, dtype=np.float))
    # Indices of examples for which we try to make predictions
    ex_inds = np.where(ex_gt_overlaps >= iou_threshold)[0] # cfg.TRAIN.BBOX_THRESH

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment_inds = ex_gt_overlaps.argmax(axis=1)
    gt_assignment_rois = gt_rois[gt_assignment_inds, :]

    regression_targets = bbox_transform(proposals[ex_inds], gt_assignment_rois[ex_inds])

    # Optionally normalize targets by a precomputed mean and stdev
    if normalize_means is not None:
        regression_targets = (regression_targets - normalize_means) / normalize_stds

    targets = np.zeros((proposals.shape[0], 6), dtype=np.float32)
    targets[ex_inds, :4] = regression_targets
    targets[ex_inds, 4] = gt_rois[gt_assignment_inds[ex_inds], 4]
    targets[ex_inds, 5] = 1 # bbiw
    return targets

class ProposalProvider:
    def __init__(self, proposal_list, proposal_cfg=None, requires_scaling=True):
        self._proposal_dict = {} if proposal_list is None else {k:v for k, v in enumerate(proposal_list)}
        self._proposal_cfg = proposal_cfg
        self._requires_scaling = requires_scaling

    @classmethod
    def fromfile(cls, filename, max_num_proposals):
        print('Reading proposals from file ({}) ...'.format(filename))
        with open(filename) as f:
            lines = f.readlines()

        proposal_list = [[] for _ in lines]
        index = 0
        cut_counter = 0
        for line in lines:
            # parse line
            numbers = line[line.find('|') + 11:]
            parsed_numbers = np.fromstring(numbers, dtype=int, sep=' ')
            parsed_rects = parsed_numbers.reshape((int(parsed_numbers.shape[0] / 4), 4))
            num_rects = parsed_rects.shape[0]
            if num_rects > max_num_proposals:
                rects = parsed_rects[:max_num_proposals,:]
                cut_counter += 1
            else:
                pad_rects = np.zeros((max_num_proposals - num_rects, 4))
                rects = np.vstack([parsed_rects, pad_rects])
            proposal_list[index] = rects
            index += 1

        print('Done. {} images had more than {} proposals.'.format(cut_counter, max_num_proposals))
        return cls(proposal_list)

    @classmethod
    def fromconfig(cls, proposal_cfg):
       return cls(None, proposal_cfg)

    @classmethod
    def fromlist(cls, proposal_list, requires_scaling):
       return cls(proposal_list, proposal_cfg=None, requires_scaling=requires_scaling)

    def requires_scaling(self):
        return self._requires_scaling

    def num_proposals(self):
        if self._proposal_cfg is None:
            return next(iter(self._proposal_dict.values())).shape[0]
        else:
            return self._proposal_cfg['NUM_ROI_PROPOSALS']

    def get_proposals(self, index, img=None):
        if index in self._proposal_dict:
            return self._proposal_dict[index]
        else:
            num_proposals = self._proposal_cfg.NUM_ROI_PROPOSALS
            return compute_proposals(img, num_proposals, self._proposal_cfg)

if __name__ == '__main__':
    import cv2
    image_file = os.path.join(abs_path, r"..\..\DataSets\Pascal\VOCdevkit\VOC2007\JPEGImages\000015.jpg")
    img = cv2.imread(image_file)

    num_proposals = 2000
    num_runs = 500
    proposals = compute_proposals(img, num_proposals, cfg=None)
    import time
    start = int(time.time())
    for i in range(num_runs):
        proposals = compute_proposals(img, num_proposals, cfg=None)
    total = int(time.time() - start)
    print ("time for {} proposals: {} (total time for {} runs: {}".format(num_proposals, total / (1.0 * num_runs), num_runs, total))

    assert len(proposals) == num_proposals, "{} != {}".format(len(proposals), num_proposals)
