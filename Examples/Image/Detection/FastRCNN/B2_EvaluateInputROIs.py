# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import sys, os
import numpy as np
from cntk_helpers import getFilesInDirectory, imWidthHeight, readGtAnnotation, bboxComputeOverlapVoc, Bbox, readRois
import PARAMETERS


####################################
# Parameters
####################################
subdirs = ['positive', 'testImages']

def evaluate_rois():
    p = PARAMETERS.get_parameters_for_dataset()
    overlaps = []
    roiCounts = []
    for subdir in subdirs:
        imgFilenames = getFilesInDirectory(os.path.join(p.imgDir, subdir), ".jpg")

        # loop over all iamges
        for imgIndex,imgFilename in enumerate(imgFilenames):
            if imgIndex % 20 == 0:
                print ("Processing subdir '{}', image {} of {}".format(subdir, imgIndex, len(imgFilenames)))
            # load ground truth
            imgPath = os.path.join(p.imgDir, subdir, imgFilename)
            imgWidth, imgHeight = imWidthHeight(imgPath)
            gtBoxes, gtLabels = readGtAnnotation(imgPath)
            gtBoxes = [Bbox(*rect) for rect in gtBoxes]

            # load rois and compute scale
            rois = readRois(p.roiDir, subdir, imgFilename)
            rois = [Bbox(*roi) for roi in rois]
            roiCounts.append(len(rois))

            # for each ground truth, compute if it is covered by an roi
            maxOverlap = -1
            for gtIndex, (gtLabel, gtBox) in enumerate(zip(gtLabels,gtBoxes)):
                assert (gtBox.max() <= max(imgWidth, imgHeight) and gtBox.max() >= 0)
                gtLabel = gtLabel.decode('utf-8')
                if gtLabel in p.classes[1:]:
                    for roi in rois:
                        assert (roi.max() <= max(imgWidth, imgHeight) and roi.max() >= 0)
                        overlap = bboxComputeOverlapVoc(gtBox, roi)
                        maxOverlap = max(maxOverlap, overlap)
                overlaps.append(maxOverlap)
    print ("Average number of rois per image " + str(1.0 * sum(roiCounts) / len(overlaps)))

    # compute recall at different overlaps
    overlaps = np.array(overlaps, np.float32)
    for overlapThreshold in np.linspace(0,1,11):
        recall = 1.0 * sum(overlaps >= overlapThreshold) / len(overlaps)
        print ("At threshold {:.2f}: recall = {:2.2f}".format(overlapThreshold, recall))
    return True

if __name__=='__main__':
    evaluate_rois()
