import numpy as np
from dlib import find_candidate_object_locations
from utils.rpn.bbox_transform import bbox_transform
from utils.cython_modules.cython_bbox import bbox_overlaps

random_seed = 23

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


def compute_proposals(img, num_proposals, min_w, min_h):
    all_rects = []
    min_size = min_w * min_h
    find_candidate_object_locations(img, all_rects, min_size=min_size)

    rects = []
    for k, d in enumerate(all_rects):
        w = d.right() - d.left()
        h = d.bottom() - d.top()
        if w < min_w or h < min_h:
            continue
        rects.append([d.left(), d.top(), d.right(), d.bottom()])

    np_rects = np.array(rects)
    num_rects = np_rects.shape[0]
    np.random.seed(random_seed)
    if num_rects < num_proposals:
        img_w = len(img[0])
        img_h = len(img)
        grid_proposals = compute_grid_proposals(num_proposals - len(rects), img_w, img_h, min_w, min_h)
        np_rects = np.vstack([np_rects, grid_proposals])
    elif len(rects) > num_proposals:
        keep_inds = range(num_rects)
        keep_inds = np.random.choice(keep_inds, size=num_proposals, replace=False)
        np_rects = np_rects[keep_inds]

    return np_rects

def compute_grid_proposals(num_proposals, img_w, img_h, min_w, min_h, max_w=None, max_h=None, aspect_ratios = [1.0], shuffle=True):
    min_wh = max(min_w, min_h)
    max_wh = min(img_h, img_w) / 2
    if max_w is not None: max_wh = min(max_wh, max_w)
    if max_h is not None: max_wh = min(max_wh, max_h)

    rects = []
    iter = 0
    while len(rects) < num_proposals:
        new_ar = []
        for ar in aspect_ratios:
            new_ar.append(ar * (0.9 ** iter))
            new_ar.append(ar * (1.1 ** iter))

        new_rects = _compute_grid_proposals(img_w, img_h, min_wh, max_wh, new_ar)
        take = min(num_proposals - len(rects), len(new_rects))
        new_rects = new_rects[:take]
        rects.extend(new_rects)

    np_rects = np.array(rects)
    num_rects = np_rects.shape[0]
    if shuffle and num_proposals < num_rects:
        keep_inds = range(num_rects)
        keep_inds = np.random.choice(keep_inds, size=num_proposals, replace=False)
        np_rects = np_rects[keep_inds]
    else:
        np_rects = np_rects[:num_proposals]

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
    def fromfile(cls, filename):
        with open(filename) as f:
            lines = f.readlines()

        proposal_list = [[] for _ in lines]
        for line in lines:
            # TODO: parse line
            index = 0
            rects = np.zeros((4, 200))
            proposal_list[index] = rects

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
        #import pdb; pdb.set_trace()
        if index in self._proposal_dict:
            return self._proposal_dict[index]
        else:
            return self._compute_proposals(img)

    def _compute_proposals(self, img):
        min_w = self._proposal_cfg['PROPOSALS_MIN_W']
        min_h = self._proposal_cfg['PROPOSALS_MIN_H']
        num_proposals = self._proposal_cfg.NUM_ROI_PROPOSALS
        return compute_proposals(img, num_proposals, min_w, min_h)

if __name__ == '__main__':
    import cv2
    image_file = r"C:\src\CNTK\Examples\Image\DataSets\Pascal\VOCdevkit\VOC2007\JPEGImages\000015.jpg"
    img = cv2.imread(image_file)

    # 0.18 sec for 4000
    # 0.15 sec for 2000
    # 0.13 sec for 1000
    num_proposals = 2000
    num_runs = 100
    import time
    start = int(time.time())
    for i in range(num_runs):
        proposals = compute_proposals(img, num_proposals, 20, 20)
    total = int(time.time() - start)
    print ("time: {}".format(total / (1.0 * num_runs)))

    assert len(proposals) == num_proposals, "{} != {}".format(len(proposals), num_proposals)
