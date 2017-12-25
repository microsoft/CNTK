# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
from builtins import chr
import os, sys, importlib, shutil
from cntk_helpers import *


####################################
# Parameters
####################################
imgDir = "C:/Users/chazhang/Desktop/newImgs/"

# no need to change these params
drawingImgSize = 1000.0


####################################
# Functions
####################################
def event_cv2_drawRectangles(event, x, y, flags, param):
    global global_image
    global global_bboxes
    global global_leftButtonDownPoint

    # draw all previous bounding boxes, and the most recent box in a different color
    imgCopy = global_image.copy()
    drawRectangles(imgCopy, global_bboxes)
    if len(global_bboxes)>0:
        drawRectangles(imgCopy, [global_bboxes[-1]], color = (255, 0, 0))

    # handle mouse events
    if event == cv2.EVENT_LBUTTONDOWN:
        global_leftButtonDownPoint = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        pt1 = global_leftButtonDownPoint
        pt2 = (x, y)
        minPt = (min(pt1[0], pt2[0]), min(pt1[1], pt2[1]))
        maxPt = (max(pt1[0], pt2[0]), max(pt1[1], pt2[1]))
        imgWidth, imgHeight = imArrayWidthHeight(global_image)
        minPt = ptClip(minPt, imgWidth, imgHeight)
        maxPt = ptClip(maxPt, imgWidth, imgHeight)
        global_bboxes.append(minPt + maxPt)

    elif flags == cv2.EVENT_FLAG_LBUTTON: #if left mouse button is held down
        cv2.rectangle(imgCopy, global_leftButtonDownPoint, (x, y), (255, 255, 0), 1)

    else:
        drawCrossbar(imgCopy, (x, y))
    cv2.imshow("AnnotationWindow", imgCopy)

def scaleCropBboxes(rectsIn, scaleFactor, imgWidth, imgHeight):
    if len(rectsIn) <= 0:
        return rectsIn
    else:
        rects = [ [int(round(rect[i]/scaleFactor)) for i in range(4)]
                  for rect in rectsIn]
        rects = [Bbox(*rect).crop(imgWidth, imgHeight).rect() for rect in rects]
        for rect in rects:
            assert (Bbox(*rect).isValid())
        return rects


####################################
# Main
####################################
imgFilenames = [f for f in os.listdir(imgDir) if f.lower().endswith(".jpg")]

# loop over each image and get annotation
for imgFilenameIndex,imgFilename in enumerate(imgFilenames):
    print (imgFilenameIndex, imgFilename)
    imgPath = os.path.join(imgDir, imgFilename)
    bBoxPath = imgPath[:-4] + ".bboxes.tsv"

    # skip image if ground truth already exists
    if os.path.exists(bBoxPath):
        print ("Skipping image {0} since ground truth already exists".format(imgFilename))
        continue
    else:
        print ("Processing image {0} of {1}: {2}".format(imgFilenameIndex, len(imgFilenames), imgPath))

    # prepare image window and callback
    global_bboxes = []
    global_image, scaleFactor = imresizeMaxDim(imread(imgPath), drawingImgSize)
    cv2.namedWindow("AnnotationWindow")
    cv2.setMouseCallback("AnnotationWindow", event_cv2_drawRectangles)
    cv2.imshow("AnnotationWindow", global_image)

    # process user input
    while True:
        key = chr(cv2.waitKey())

        # undo/remove last rectangle
        if key == "u":
            if len(global_bboxes) >= 1:
                global_bboxes = global_bboxes[:-1]
                imgCopy = global_image.copy()
                drawRectangles(imgCopy, global_bboxes)
                cv2.imshow("AnnotationWindow", imgCopy)

        # skip image
        elif key == "s":
            if os.path.exists(bBoxPath):
                print ("Skipping image hence deleting existing bbox file: " + bBoxPath)
                os.remove(bBoxPath)
            break

        # next image
        elif key == "n":
            bboxes = scaleCropBboxes(global_bboxes, scaleFactor, imWidth(imgPath), imHeight(imgPath))
            writeTable(bBoxPath, bboxes)
            break

        # quit
        elif key == "q":
            sys.exit()

cv2.destroyAllWindows()
