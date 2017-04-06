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
train = False
trainSubDirs = ['positive', 'negative'] # , 'testImages'
testSubDirs = ['testImages']

# no need to change these parameters
boAddSelectiveSearchROIs = True
boAddRoisOnGrid = True


if not datasetName.startswith("pascalVoc"):
    # init
    makeDirectory(roiDir)
    roi_minDim = roi_minDimRel * roi_maxImgDim
    roi_maxDim = roi_maxDimRel * roi_maxImgDim
    roi_minNrPixels = roi_minNrPixelsRel * roi_maxImgDim*roi_maxImgDim
    roi_maxNrPixels = roi_maxNrPixelsRel * roi_maxImgDim*roi_maxImgDim

    if train:
        roiPath = "{}{}.txt".format(roiDir, "train.GTRois")
        mapPath = "{}{}.txt".format(roiDir, "train.imgMap")
        sudirs = trainSubDirs
    else:
        roiPath = "{}{}.txt".format(roiDir, "test.GTRois")
        mapPath = "{}{}.txt".format(roiDir, "test.imgMap")
        sudirs = testSubDirs

    print(classes)
    print("writing to file {}".format(roiPath))
    cnt = 0
    with open(roiPath, 'w') as roisFile, \
         open(mapPath, 'w') as mapFile:
            for subdir in sudirs:
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

                    # TODO: return scaled and padded boxes?
                    # For CNTK: convert and scale gt_box coords from x, y, w, h relative to x1, y1, x2, y2 absolute
                    #whwh = (1000, 1000, 1000, 1000)  # TODO: get image width and height OR better scale beforehand
                    #ngtb = np.vstack((gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 0] + gt_boxes[:, 2], gt_boxes[:, 1] + gt_boxes[:, 3]))
                    #gt_boxes[:, :-1] = ngtb.transpose() * whwh

                    boxesStr = "|roiAndLabel "
                    for boxIndex, box in enumerate(gtBoxes):
                        rect = roiTransformPadScale(box, w_offset, h_offset, scale)
                        label = gtLabelInd[boxIndex]
                        if label > 0:
                            boxesStr += getCntkRoiCoordsLine(rect, cntk_padWidth, cntk_padHeight)
                            boxesStr += " {}".format(label)

                    if boxesStr != "|roiAndLabel ":
                        roisFile.write("{}\t{}\n".format(cnt, boxesStr))
                        mapFile.write("{}\t{}\t0\n".format(cnt, imgPath))
                        cnt += 1

print ("DONE.")
