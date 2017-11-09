# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os, importlib, sys
from cntk_helpers import imWidthHeight, nnPredict, applyNonMaximaSuppression, makeDirectory, visualizeResults, imshow
import PARAMETERS

####################################
# Parameters
####################################
image_set = 'test'      # 'train', 'test'

def visualize_output_rois(testing=False):
    p = PARAMETERS.get_parameters_for_dataset()

    # no need to change these parameters
    boUseNonMaximaSurpression = True
    visualizationDir = os.path.join(p.resultsDir, "visualizations")
    cntkParsedOutputDir = os.path.join(p.cntkFilesDir, image_set + "_parsed")

    makeDirectory(p.resultsDir)
    makeDirectory(visualizationDir)

    # loop over all images and visualize
    imdb = p.imdbs[image_set]
    for imgIndex in range(0, imdb.num_images):
        imgPath = imdb.image_path_at(imgIndex)
        imgWidth, imgHeight = imWidthHeight(imgPath)

        # evaluate classifier for all rois
        labels, scores = nnPredict(imgIndex, cntkParsedOutputDir, p.cntk_nrRois, len(p.classes), None)

        # remove the zero-padded rois
        scores = scores[:len(imdb.roidb[imgIndex]['boxes'])]
        labels = labels[:len(imdb.roidb[imgIndex]['boxes'])]

        # perform non-maxima surpression. note that the detected classes in the image is not affected by this.
        nmsKeepIndices = []
        if boUseNonMaximaSurpression:
            nmsKeepIndices = applyNonMaximaSuppression(p.nmsThreshold, labels, scores, imdb.roidb[imgIndex]['boxes'])
            print ("Non-maxima surpression kept {:4} of {:4} rois (nmsThreshold={})".format(len(nmsKeepIndices), len(labels), p.nmsThreshold))

        # visualize results
        imgDebug = visualizeResults(imgPath, labels, scores, imdb.roidb[imgIndex]['boxes'], p.cntk_padWidth, p.cntk_padHeight,
                                    p.classes, nmsKeepIndices, boDrawNegativeRois=True)
        if not testing:
            imshow(imgDebug, waitDuration=0, maxDim = 800)
            # imwrite(imgDebug, visualizationDir + "/" + str(imgIndex) + os.path.basename(imgPath))

    print ("DONE.")
    return True

if __name__=='__main__':
    visualize_output_rois()
