import os, importlib, sys
from cntk_helpers import *
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)


####################################
# Parameters
####################################
image_set = 'train' # 'train', 'test'

# no need to change these parameters
parseNrImages = 10 # for speed reasons only parse CNTK file for the first N images
boUseNonMaximaSurpression = False
nmsThreshold = 0.1


####################################
# Main
####################################

print "Load ROI co-ordinates and labels"
cntkImgsPath, cntkRoiCoordsPath, cntkRoiLabelsPath, nrRoisPath = getCntkInputPaths(cntkFilesDir, image_set)
imgPaths = getColumn(readTable(cntkImgsPath), 1)
nrRealRois = [int(s) for s in readFile(nrRoisPath)]
roiAllLabels = readCntkRoiLabels(cntkRoiLabelsPath, cntk_nrRois, len(classes), parseNrImages)
if parseNrImages:
    imgPaths = imgPaths[:parseNrImages]
    nrRealRois = nrRealRois[:parseNrImages]
    roiAllLabels = roiAllLabels[:parseNrImages]
roiAllCoords = readCntkRoiCoordinates(imgPaths, cntkRoiCoordsPath, cntk_nrRois, cntk_padWidth, cntk_padHeight,
                                      parseNrImages)
assert (len(imgPaths) == len(roiAllCoords) == len(roiAllLabels) == len(nrRealRois))

# loop over all images and visualize
for imgIndex, imgPath in enumerate(imgPaths):
    print "Visualizing image %d at %s..." % (imgIndex, imgPath)
    roiCoords = roiAllCoords[imgIndex][:nrRealRois[imgIndex]]
    roiLabels = roiAllLabels[imgIndex][:nrRealRois[imgIndex]]

    # perform non-maxima surpression. note that the detected classes in the image is not affected by this.
    nmsKeepIndices = []
    if boUseNonMaximaSurpression:
        imgWidth, imgHeight = imWidthHeight(imgPath)
        nmsKeepIndices = applyNonMaximaSuppression(nmsThreshold, roiLabels, [0] * len(roiLabels), roiCoords)
        print "Non-maxima surpression kept {} of {} rois (nmsThreshold={})".format(len(nmsKeepIndices),
                                                                                   len(roiLabels), nmsThreshold)

    # visualize results
    imgDebug = visualizeResults(imgPath, roiLabels, None, roiCoords, cntk_padWidth, cntk_padHeight,
                                classes, nmsKeepIndices, boDrawNegativeRois=True)
    imshow(imgDebug, waitDuration=0, maxDim=800)

print "DONE."
