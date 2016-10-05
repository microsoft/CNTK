# -*- coding: utf-8 -*-
import sys, os, importlib
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)


####################################
# Parameters
####################################
subdirs = ['positive', 'testImages']


####################################
# Main
####################################
overlaps = []
roiCounts = []
for subdir in subdirs:
    imgFilenames = getFilesInDirectory(imgDir + subdir, ".jpg")

    # loop over all iamges
    for imgIndex,imgFilename in enumerate(imgFilenames):
        if imgIndex % 20 == 0:
            print "Processing subdir '{}', image {} of {}".format(subdir, imgIndex, len(imgFilenames))
        # load ground truth
        imgPath = imgDir + subdir + "/" + imgFilename
        imgWidth, imgHeight = imWidthHeight(imgPath)
        gtBoxes, gtLabels = readGtAnnotation(imgPath)
        gtBoxes = [Bbox(*rect) for rect in gtBoxes]

        # load rois and compute scale
        rois = readRois(roiDir, subdir, imgFilename)
        rois = [Bbox(*roi) for roi in rois]
        roiCounts.append(len(rois))

        # for each ground truth, compute if it is covered by an roi
        maxOverlap = -1
        for gtIndex, (gtLabel, gtBox) in enumerate(zip(gtLabels,gtBoxes)):
            assert (gtBox.max() <= max(imgWidth, imgHeight) and gtBox.max() >= 0)
            if gtLabel in classes[1:]:
                for roi in rois:
                    assert (roi.max() <= max(imgWidth, imgHeight) and roi.max() >= 0)
                    overlap = bboxComputeOverlapVoc(gtBox, roi)
                    maxOverlap = max(maxOverlap, overlap)
        overlaps.append(maxOverlap)
print "Average number of rois per image " + str(1.0 * sum(roiCounts) / len(overlaps))

# compute recall at different overlaps
overlaps = np.array(overlaps, np.float32)
for overlapThreshold in np.linspace(0,1,11):
    recall = 1.0 * sum(overlaps >= overlapThreshold) / len(overlaps)
    print "At threshold {:.2f}: recall = {:2.2f}".format(overlapThreshold, recall)
