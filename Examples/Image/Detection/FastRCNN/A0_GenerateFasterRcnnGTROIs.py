from __future__ import print_function
from builtins import input
import os, sys, importlib
import shutil, time
from cntk_helpers import readGtAnnotation
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)


####################################
# Parameters
####################################
boSaveDebugImg = True
trainSubDirs = ['positive', 'negative'] # , 'testImages'
image_sets = ["train", "test"]

# no need to change these parameters
boAddSelectiveSearchROIs = True
boAddRoisOnGrid = True


####################################
# Main
####################################
# generate ROIs using selective search and grid (for pascal we use the precomputed ROIs from Ross)
if not datasetName.startswith("pascalVoc"):
    # init
    makeDirectory(roiDir)
    roi_minDim = roi_minDimRel * roi_maxImgDim
    roi_maxDim = roi_maxDimRel * roi_maxImgDim
    roi_minNrPixels = roi_minNrPixelsRel * roi_maxImgDim*roi_maxImgDim
    roi_maxNrPixels = roi_maxNrPixelsRel * roi_maxImgDim*roi_maxImgDim

    roiPath = "{}{}.txt".format(roiDir, "train.GTRois")
    mapPath = "{}{}.txt".format(roiDir, "train.imgMap")
    print(classes)
    print("writing to file {}".format(roiPath))
    cnt = 0
    with open(roiPath, 'w') as roisFile, \
         open(mapPath, 'w') as mapFile:
            for subdir in trainSubDirs:
                makeDirectory(roiDir + subdir)
                imgFilenames = getFilesInDirectory(os.path.join(imgDir, subdir), ".jpg")

                # loop over all images
                for imgIndex,imgFilename in enumerate(imgFilenames):

                    # load image
                    print (imgIndex, len(imgFilenames), subdir, imgFilename)
                    imgPath = os.path.normpath(os.path.join(imgDir, subdir, imgFilename)).replace('\\', '/')

                    gtBoxes, gtLabels = readGtAnnotation(imgPath)
                    labels = [l.decode('utf-8') for l in gtLabels]
                    class_to_ind = dict(zip(classes, range(len(classes) + 1)))
                    gtLabelInd = [class_to_ind[gtl] for gtl in labels]

                    # all rois need to be scaled + padded to cntk input image size
                    imgWidth, imgHeight = imWidthHeight(imgPath)
                    targetw, targeth, w_offset, h_offset, scale = roiTransformPadScaleParams(
                        imgWidth, imgHeight, cntk_padWidth, cntk_padHeight)

                    boxesStr = "|roiAndLabel "
                    for boxIndex, box in enumerate(gtBoxes):
                        rect = roiTransformPadScale(box, w_offset, h_offset, scale)
                        boxesStr += getCntkRoiCoordsLine(rect, cntk_padWidth, cntk_padHeight)
                        boxesStr += " {}".format(gtLabelInd[boxIndex])

                    roisFile.write("{}\t{}\n".format(cnt, boxesStr))
                    mapFile.write("{}]\t{}\t0\n".format(cnt, imgPath))
                    cnt += 1

print ("DONE.")
