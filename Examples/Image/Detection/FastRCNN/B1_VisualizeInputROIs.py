# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os, importlib, sys
from cntk_helpers import *
import PARAMETERS


####################################
# Parameters
####################################
image_set = 'train' # 'train', 'test'

# no need to change these parameters
parseNrImages = 10 # for speed reasons only parse CNTK file for the first N images
boUseNonMaximaSurpression = False
nmsThreshold = 0.1


def generate_rois_visualization(testing=False):
    p = PARAMETERS.get_parameters_for_dataset()

    print ("Load ROI co-ordinates and labels")
    cntkImgsPath, cntkRoiCoordsPath, cntkRoiLabelsPath, nrRoisPath = getCntkInputPaths(p.cntkFilesDir, image_set)
    imgPaths = getColumn(readTable(cntkImgsPath), 1)
    nrRealRois = [int(s) for s in readFile(nrRoisPath)]
    roiAllLabels = readCntkRoiLabels(cntkRoiLabelsPath, p.cntk_nrRois, len(p.classes), parseNrImages)
    if parseNrImages:
        imgPaths = imgPaths[:parseNrImages]
        nrRealRois = nrRealRois[:parseNrImages]
        roiAllLabels = roiAllLabels[:parseNrImages]
    roiAllCoords = readCntkRoiCoordinates(imgPaths, cntkRoiCoordsPath, p.cntk_nrRois, p.cntk_padWidth, p.cntk_padHeight,
                                          parseNrImages)
    assert (len(imgPaths) == len(roiAllCoords) == len(roiAllLabels) == len(nrRealRois))

    # loop over all images and visualize
    for imgIndex, imgPath in enumerate(imgPaths):
        print ("Visualizing image %d at %s..." % (imgIndex, imgPath))
        roiCoords = roiAllCoords[imgIndex][:nrRealRois[imgIndex]]
        roiLabels = roiAllLabels[imgIndex][:nrRealRois[imgIndex]]

        # perform non-maxima surpression. note that the detected classes in the image is not affected by this.
        nmsKeepIndices = []
        if boUseNonMaximaSurpression:
            imgWidth, imgHeight = imWidthHeight(imgPath)
            nmsKeepIndices = applyNonMaximaSuppression(nmsThreshold, roiLabels, [0] * len(roiLabels), roiCoords)
            print ("Non-maxima surpression kept {} of {} rois (nmsThreshold={})".format(len(nmsKeepIndices),
                                                                                       len(roiLabels), nmsThreshold))

        # visualize results
        imgDebug = visualizeResults(imgPath, roiLabels, None, roiCoords, p.cntk_padWidth, p.cntk_padHeight,
                                    p.classes, nmsKeepIndices, boDrawNegativeRois=True)
        if not testing:
            imshow(imgDebug, waitDuration=0, maxDim=800)

    print ("DONE.")
    return True

if __name__=='__main__':
    generate_rois_visualization()
