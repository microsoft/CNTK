#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Train post-hoc SVMs using the algorithm and hyper-parameters from
traditional R-CNN.
"""

from timer import Timer
from sklearn import svm
import numpy as np



#################################################
# Slightly modified SVM training functions
#################################################
class SVMTrainer(object):
    """
    Trains post-hoc detection SVMs for all classes using the algorithm
    and hyper-parameters of traditional R-CNN.
    """

    def __init__(self, net, imdb, im_detect, svmWeightsPath, svmBiasPath, svmFeatScalePath,
                 svm_C, svm_B, svm_nrEpochs, svm_retrainLimit, svm_evictThreshold, svm_posWeight,
                 svm_targetNorm, svm_penality, svm_loss, svm_rngSeed):
        self.net = net
        self.imdb = imdb
        self.im_detect = im_detect
        self.svm_nrEpochs = svm_nrEpochs
        self.svm_targetNorm = svm_targetNorm
        self.svmWeightsPath = svmWeightsPath
        self.svmBiasPath = svmBiasPath
        self.svmFeatScalePath = svmFeatScalePath
        self.layer = 'fc7'
        self.hard_thresh = -1.0001
        self.neg_iou_thresh = 0.3
        dim = net.params['cls_score'][0].data.shape[1]
        self.feature_scale = self._get_feature_scale()
        print('Feature dim: {}'.format(dim))
        print('Feature scale: {:.3f}'.format(self.feature_scale))
        self.trainers = [SVMClassTrainer(cls, dim, self.feature_scale, svm_C, svm_B, svm_posWeight, svm_penality, svm_loss,
                                         svm_rngSeed, svm_retrainLimit, svm_evictThreshold) for cls in imdb.classes]


    def _get_feature_scale(self, num_images=100):
        _t = Timer()
        roidb = self.imdb.roidb
        total_norm = 0.0
        total_sum = 0.0
        count = 0.0
        num_images = min(num_images, self.imdb.num_images)
        inds = np.random.choice(xrange(self.imdb.num_images), size=num_images, replace=False)

        for i_, i in enumerate(inds):
            #im = cv2.imread(self.imdb.image_path_at(i))
            #if roidb[i]['flipped']:
            #    im = im[:, ::-1, :]
            #im = self.imdb.image_path_at(i)
            _t.tic()
            scores, boxes, feat = self.im_detect(self.net, i, roidb[i]['boxes'], boReturnClassifierScore = False)
            _t.toc()
            #feat = self.net.blobs[self.layer].data
            total_norm += np.sqrt((feat ** 2).sum(axis=1)).sum()
            total_sum += 1.0 * sum(sum(feat)) / len(feat)
            count += feat.shape[0]
            print('{}/{}: avg feature norm: {:.3f}, average value: {:.3f}'.format(i_ + 1, num_images,
                                                           total_norm / count, total_sum / count))

        return self.svm_targetNorm * 1.0 / (total_norm / count)

    def _get_pos_counts(self):
        counts = np.zeros((len(self.imdb.classes)), dtype=np.int)
        roidb = self.imdb.roidb
        for i in xrange(len(roidb)):
            for j in xrange(1, self.imdb.num_classes):
                I = np.where(roidb[i]['gt_classes'] == j)[0]
                counts[j] += len(I)

        for j in xrange(1, self.imdb.num_classes):
            print('class {:s} has {:d} positives'.
                  format(self.imdb.classes[j], counts[j]))

        return counts

    def get_pos_examples(self):
        counts = self._get_pos_counts()
        for i in xrange(len(counts)):
            self.trainers[i].alloc_pos(counts[i])

        _t = Timer()
        roidb = self.imdb.roidb
        num_images = len(roidb)
        for i in xrange(num_images):
            #im = cv2.imread(self.imdb.image_path_at(i))
            #if roidb[i]['flipped']:
            #    im = im[:, ::-1, :]
            #im = self.imdb.image_path_at(i)
            gt_inds = np.where(roidb[i]['gt_classes'] > 0)[0]
            gt_boxes = roidb[i]['boxes'][gt_inds]
            _t.tic()
            scores, boxes, feat = self.im_detect(self.net, i, gt_boxes, self.feature_scale, gt_inds, boReturnClassifierScore = False)
            _t.toc()
            #feat = self.net.blobs[self.layer].data
            for j in xrange(1, self.imdb.num_classes):
                cls_inds = np.where(roidb[i]['gt_classes'][gt_inds] == j)[0]
                if len(cls_inds) > 0:
                    cls_feat = feat[cls_inds, :]
                    self.trainers[j].append_pos(cls_feat)
            if i % 50 == 0:
                print 'get_pos_examples: {:d}/{:d} {:.3f}s' \
                      .format(i + 1, len(roidb), _t.average_time)

    def initialize_net(self):
        # Start all SVM parameters at zero
        self.net.params['cls_score'][0].data[...] = 0
        self.net.params['cls_score'][1].data[...] = 0

        # Initialize SVMs in a smart way. Not doing this because its such
        # a good initialization that we might not learn something close to
        # the SVM solution.
    #        # subtract background weights and biases for the foreground classes
    #        w_bg = self.net.params['cls_score'][0].data[0, :]
    #        b_bg = self.net.params['cls_score'][1].data[0]
    #        self.net.params['cls_score'][0].data[1:, :] -= w_bg
    #        self.net.params['cls_score'][1].data[1:] -= b_bg
    #        # set the background weights and biases to 0 (where they shall remain)
    #        self.net.params['cls_score'][0].data[0, :] = 0
    #        self.net.params['cls_score'][1].data[0] = 0

    def update_net(self, cls_ind, w, b):
        self.net.params['cls_score'][0].data[cls_ind, :] = w
        self.net.params['cls_score'][1].data[cls_ind] = b

    def train_with_hard_negatives(self):
        _t = Timer()
        roidb = self.imdb.roidb
        num_images = len(roidb)

        for epoch in range(0,self.svm_nrEpochs):

            # num_images = 100
            for i in xrange(num_images):
                print "*** EPOCH = %d, IMAGE = %d *** " % (epoch, i)
                #im = cv2.imread(self.imdb.image_path_at(i))
                #if roidb[i]['flipped']:
                #    im = im[:, ::-1, :]
                #im = self.imdb.image_path_at(i)
                _t.tic()
                scores, boxes, feat = self.im_detect(self.net, i, roidb[i]['boxes'], self.feature_scale)
                _t.toc()
                #feat = self.net.blobs[self.layer].data
                for j in xrange(1, self.imdb.num_classes):
                    hard_inds = \
                        np.where((scores[:, j] > self.hard_thresh) &
                                 (roidb[i]['gt_overlaps'][:, j].toarray().ravel() <
                                  self.neg_iou_thresh))[0]
                    if len(hard_inds) > 0:
                        hard_feat = feat[hard_inds, :].copy()
                        new_w_b = \
                            self.trainers[j].append_neg_and_retrain(feat=hard_feat)
                        if new_w_b is not None:
                            self.update_net(j, new_w_b[0], new_w_b[1])
                            np.savetxt(self.svmWeightsPath[:-4]   + "_epoch" + str(epoch) + ".txt", self.net.params['cls_score'][0].data)
                            np.savetxt(self.svmBiasPath[:-4]      + "_epoch" + str(epoch) + ".txt", self.net.params['cls_score'][1].data)
                            np.savetxt(self.svmFeatScalePath[:-4] + "_epoch" + str(epoch) + ".txt", [self.feature_scale])

            print(('train_with_hard_negatives: '
                   '{:d}/{:d} {:.3f}s').format(i + 1, len(roidb),
                                               _t.average_time))

    def train(self):
        # Initialize SVMs using
        #   a. w_i = fc8_w_i - fc8_w_0
        #   b. b_i = fc8_b_i - fc8_b_0
        #   c. Install SVMs into net
        self.initialize_net()

        # Pass over roidb to count num positives for each class
        #   a. Pre-allocate arrays for positive feature vectors
        # Pass over roidb, computing features for positives only
        self.get_pos_examples()

        # Pass over roidb
        #   a. Compute cls_score with forward pass
        #   b. For each class
        #       i. Select hard negatives
        #       ii. Add them to cache
        #   c. For each class
        #       i. If SVM retrain criteria met, update SVM
        #       ii. Install new SVM into net
        self.train_with_hard_negatives()

        # One final SVM retraining for each class
        # Install SVMs into net
        for j in xrange(1, self.imdb.num_classes):
            new_w_b = self.trainers[j].append_neg_and_retrain(force=True)
            self.update_net(j, new_w_b[0], new_w_b[1])

        #save svm
        np.savetxt(self.svmWeightsPath,   self.net.params['cls_score'][0].data)
        np.savetxt(self.svmBiasPath,      self.net.params['cls_score'][1].data)
        np.savetxt(self.svmFeatScalePath, [self.feature_scale])


class SVMClassTrainer(object):
    """Manages post-hoc SVM training for a single object class."""

    def __init__(self, cls, dim, feature_scale,
                 C, B, pos_weight, svm_penality, svm_loss, svm_rngSeed, svm_retrainLimit, svm_evictThreshold):
        self.pos = np.zeros((0, dim), dtype=np.float32)
        self.neg = np.zeros((0, dim), dtype=np.float32)
        self.B = B
        self.C = C
        self.cls = cls
        self.pos_weight = pos_weight
        self.dim = dim
        self.feature_scale = feature_scale
        if type(pos_weight) == str:  #e.g. pos_weight == 'auto'
            class_weight = pos_weight
        else:
            class_weight = {1: pos_weight, -1: 1}

        self.svm = svm.LinearSVC(C=C, class_weight=class_weight,
                                 intercept_scaling=B, verbose=1,
                                 penalty=svm_penality, loss=svm_loss,
                                 random_state=svm_rngSeed, dual=True)

        self.pos_cur = 0
        self.num_neg_added = 0
        self.retrain_limit = svm_retrainLimit
        self.evict_thresh = svm_evictThreshold
        self.loss_history = []

    def alloc_pos(self, count):
        self.pos_cur = 0
        self.pos = np.zeros((count, self.dim), dtype=np.float32)

    def append_pos(self, feat):
        num = feat.shape[0]
        self.pos[self.pos_cur:self.pos_cur + num, :] = feat
        self.pos_cur += num

    def train(self):
        print('>>> Updating {} detector <<<'.format(self.cls))
        num_pos = self.pos.shape[0]
        num_neg = self.neg.shape[0]
        print('Cache holds {} pos examples and {} neg examples'.
              format(num_pos, num_neg))
        X = np.vstack((self.pos, self.neg)) * self.feature_scale
        y = np.hstack((np.ones(num_pos),
                       -np.ones(num_neg)))
        self.svm.fit(X, y)
        w = self.svm.coef_
        b = self.svm.intercept_[0]

        scores = self.svm.decision_function(X)
        pos_scores = scores[:num_pos]
        neg_scores = scores[num_pos:]

        num_neg_wrong = sum(neg_scores > 0)
        num_pos_wrong = sum(pos_scores < 0)
        meanAcc = 0.5 * (num_pos - num_pos_wrong) / num_pos + 0.5*(num_neg - num_neg_wrong) / num_neg
        if type(self.pos_weight) == str:
            pos_loss = 0
        else:
            pos_loss = (self.C * self.pos_weight *
                        np.maximum(0, 1 - pos_scores).sum())
        neg_loss = self.C * np.maximum(0, 1 + neg_scores).sum()
        reg_loss = 0.5 * np.dot(w.ravel(), w.ravel()) + 0.5 * b ** 2
        tot_loss = pos_loss + neg_loss + reg_loss
        self.loss_history.append((meanAcc, num_pos_wrong, num_pos, num_neg_wrong, num_neg, tot_loss, pos_loss, neg_loss, reg_loss))
        for i, losses in enumerate(self.loss_history):
            print(('    {:4d}: meanAcc={:.3f} -- pos wrong: {:5}/{:5}; neg wrong: {:5}/{:5};  '
                   '     obj val: {:.3f} = {:.3f}  (posUnscaled) + {:.3f} (neg) + {:.3f} (reg)').format(i, *losses))

        # Sanity check

        scores_ret = (
                         X * 1.0 / self.feature_scale).dot(w.T * self.feature_scale) + b
        assert np.allclose(scores, scores_ret[:, 0], atol=1e-5), \
                "Scores from returned model don't match decision function"

        return ((w * self.feature_scale, b), pos_scores, neg_scores)

    def append_neg_and_retrain(self, feat=None, force=False):
        if feat is not None:
            num = feat.shape[0]
            self.neg = np.vstack((self.neg, feat))
            self.num_neg_added += num
        if self.num_neg_added > self.retrain_limit or force:
            self.num_neg_added = 0
            new_w_b, pos_scores, neg_scores = self.train()
            # scores = np.dot(self.neg, new_w_b[0].T) + new_w_b[1]
            # easy_inds = np.where(neg_scores < self.evict_thresh)[0]
            print('    Pruning easy negatives')
            print('         before pruning: #neg = ' + str(len(self.neg)))
            not_easy_inds = np.where(neg_scores >= self.evict_thresh)[0]
            if len(not_easy_inds) > 0:
                self.neg = self.neg[not_easy_inds, :]
                # self.neg = np.delete(self.neg, easy_inds)
            print('         after pruning: #neg = ' + str(len(self.neg)))
            print('    Cache holds {} pos examples and {} neg examples'.
                  format(self.pos.shape[0], self.neg.shape[0]))
            print('    {} pos support vectors'.format((pos_scores <= 1).sum()))
            print('    {} neg support vectors'.format((neg_scores >= -1).sum()))
            return new_w_b
        else:
            return None
