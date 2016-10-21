# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

#from config import cfg #, get_output_dir
#from blob import im_list_to_blob
import os, cv2, numpy as np, cPickle, heapq
import nms as nmsPython
from cython_nms import nms
from timer import Timer
from cntk_helpers import im_detect, apply_nms


def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def _bbox_pred(boxes, box_deltas):
    """Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + cfg.EPS
    heights = boxes[:, 3] - boxes[:, 1] + cfg.EPS
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = box_deltas[:, 0::4]
    dy = box_deltas[:, 1::4]
    dw = box_deltas[:, 2::4]
    dh = box_deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes

# def im_detect(net, im, boxes):
#     """Detect object classes in an image given object proposals.
#
#     Arguments:
#         net (caffe.Net): Fast R-CNN network to use
#         im (ndarray): color image to test (in BGR order)
#         boxes (ndarray): R x 4 array of object proposals
#
#     Returns:
#         scores (ndarray): R x K array of object class scores (K includes
#             background as object category 0)
#         boxes (ndarray): R x (4*K) array of predicted bounding boxes
#     """
#     blobs, unused_im_scale_factors = _get_blobs(im, boxes)
#
#     # When mapping from image ROIs to feature map ROIs, there's some aliasing
#     # (some distinct image ROIs get mapped to the same feature ROI).
#     # Here, we identify duplicate feature ROIs, so we only compute features
#     # on the unique subset.
#     if cfg.DEDUP_BOXES > 0:
#         v = np.array([1, 1e3, 1e6, 1e9, 1e12])
#         hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
#         _, index, inv_index = np.unique(hashes, return_index=True,
#                                         return_inverse=True)
#         blobs['rois'] = blobs['rois'][index, :]
#         boxes = boxes[index, :]
#
#     # reshape network inputs
#     net.blobs['data'].reshape(*(blobs['data'].shape))
#     net.blobs['rois'].reshape(*(blobs['rois'].shape))
#     blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False),
#                             rois=blobs['rois'].astype(np.float32, copy=False))
#     if cfg.TEST.SVM:
#         # use the raw scores before softmax under the assumption they
#         # were trained as linear SVMs
#         scores = net.blobs['cls_score'].data
#     else:
#         # use softmax estimated probabilities
#         scores = blobs_out['cls_prob']
#
#     if cfg.TEST.BBOX_REG:
#         # Apply bounding-box regression deltas
#         box_deltas = blobs_out['bbox_pred']
#         pred_boxes = _bbox_pred(boxes, box_deltas)
#         pred_boxes = _clip_boxes(pred_boxes, im.shape)
#     else:
#         # Simply repeat the boxes, once for each class
#         pred_boxes = np.tile(boxes, (1, scores.shape[1]))
#
#     if cfg.DEDUP_BOXES > 0:
#         # Map scores and predictions back to the original set of boxes
#         scores = scores[inv_index, :]
#         pred_boxes = pred_boxes[inv_index, :]
#
#     return scores, pred_boxes

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def test_net(net, imdb, output_dir, feature_scale, classifier = 'svm', nmsThreshold = 0.3,
             boUsePythonImpl = False, boThresholdDetections = True, boApplyNms = True):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS
    max_per_set = 40 * num_images
    # heuristic: keep at most 100 detection per class per image prior to NMS
    max_per_image = 100
    # detection thresold for each class (this is adaptively set based on the
    # max_per_set constraint)
    thresh = -np.inf * np.ones(imdb.num_classes)
    # top_scores will hold one minheap of scores per class (used to enforce
    # the max_per_set constraint)
    top_scores = [[] for _ in xrange(imdb.num_classes)]
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    #output_dir = get_output_dir(imdb, net)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    roidb = imdb.roidb

    if not boThresholdDetections:
        for i in xrange(num_images):
            if i % 1000 == 0:
                print "   Processing image {} of {}..".format(i, num_images)
            scores, _, _ = im_detect(net, i, roidb[i]['boxes'], feature_scale=feature_scale, classifier=classifier)

            for j in xrange(1, imdb.num_classes):
                inds = np.where(roidb[i]['gt_classes'] == 0)[0]
                cls_scores = scores[inds, j]
                cls_boxes = roidb[i]['boxes'][inds]
                all_boxes[j][i] = \
                    np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                        .astype(np.float32, copy=False)

    else:
        for i in xrange(num_images):
            if i % 1000 == 0:
                print "   Processing image {} of {}..".format(i, num_images)
            #im = cv2.imread(imdb.image_path_at(i))
            #_t['im_detect'].tic()
            scores, _, _ = im_detect(net, i, roidb[i]['boxes'], feature_scale = feature_scale, classifier = classifier)
            #_t['im_detect'].toc()

            _t['misc'].tic()
            for j in xrange(1, imdb.num_classes):
                inds = np.where((scores[:, j] > thresh[j]) &
                                (roidb[i]['gt_classes'] == 0))[0]
                cls_scores = scores[inds, j]

                # cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
                boxes = roidb[i]['boxes']
                cls_boxes = boxes[inds]

                top_inds = np.argsort(-cls_scores)[:max_per_image]
                cls_scores = cls_scores[top_inds]
                cls_boxes = cls_boxes[top_inds, :]
                # push new scores onto the minheap
                for val in cls_scores:
                    heapq.heappush(top_scores[j], val)
                # if we've collected more than the max number of detection,
                # then pop items off the minheap and update the class threshold
                if len(top_scores[j]) > max_per_set:
                    while len(top_scores[j]) > max_per_set:
                        heapq.heappop(top_scores[j])
                    thresh[j] = top_scores[j][0]

                all_boxes[j][i] = \
                        np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                        .astype(np.float32, copy=False)

                #visualize rois
                if False:
                    im = cv2.imread(imdb.image_path_at(i))
                    if boUsePythonImpl:
                        keep = nmsPython.nms(all_boxes[j][i], 0.3)
                    else:
                        keep = nms(all_boxes[j][i], 0.3)
                    vis_detections(im, imdb.classes[j], all_boxes[j][i][keep, :])
            _t['misc'].toc()

            # print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
            #       .format(i + 1, num_images, _t['im_detect'].average_time,
            #               _t['misc'].average_time)

        #keep only the boxes with highest score for each class
        #   shape of all_boxes: e.g. 21 classes x 4952 images x 58 rois x 5 coord+score
        for j in xrange(1, imdb.num_classes):
            for i in xrange(num_images):
                inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
                if len(inds) == 0:
                    all_boxes[j][i] = []
                else:
                    all_boxes[j][i] = all_boxes[j][i][inds, :]

    if output_dir:
        det_file = os.path.join(output_dir, 'detections.pkl')
        with open(det_file, 'wb') as f:
            cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    if boApplyNms:
        print "Number of rois before non-maxima surpression: %d" % sum([len(all_boxes[i][j]) for i in range(imdb.num_classes) for j in range(imdb.num_images)])
        nms_dets,_ = apply_nms(all_boxes, nmsThreshold, boUsePythonImpl)
        print "Number of rois  after non-maxima surpression: %d" % sum([len(nms_dets[i][j]) for i in range(imdb.num_classes) for j in range(imdb.num_images)])
    else:
        print "Skipping non-maxima surpression"
        nms_dets = all_boxes

    print 'Evaluating detections'
    imdb.evaluate_detections(nms_dets, output_dir)
