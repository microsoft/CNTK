import pdb, sys, os, time
import numpy as np
import selectivesearch
from easydict import EasyDict
from fastRCNN.nms import nms as nmsPython




####################################
# Region-of-interest
####################################
def getSelectiveSearchRois(img, ssScale, ssSigma, ssMinSize, maxDim):
    # Selective Search
    #     Parameters
    #     ----------
    #         im_orig : ndarray
    #             Input image
    #         scale : int
    #             Free parameter. Higher means larger clusters in felzenszwalb segmentation.
    #         sigma : float
    #             Width of Gaussian kernel for felzenszwalb segmentation.
    #         min_size : int
    #             Minimum component size for felzenszwalb segmentation.
    #     Returns
    #     -------
    #         img : ndarray
    #             image with region label
    #             region label is stored in the 4th value of each pixel [r,g,b,(region)]
    #         regions : array of dict
    #             [
    #                 {
    #                     'rect': (left, top, right, bottom),
    #                     'labels': [...]
    #                 },
    #                 ...
    #             ]
    #inter_area seems to give much better results esp when upscaling image
    img, scale = imresizeMaxDim(img, maxDim, boUpscale=True, interpolation = cv2.INTER_AREA)
    _, ssRois = selectivesearch.selective_search(img, scale=ssScale, sigma=ssSigma, min_size=ssMinSize)
    rects = []
    for ssRoi in ssRois:
        x, y, w, h = ssRoi['rect']
        rects.append([x,y,x+w,y+h])
    return rects, img, scale


def getGridRois(imgWidth, imgHeight, nrGridScales, aspectRatios = [1.0]):
    rects = []
    #start adding large ROIs and then smaller ones
    for iter in range(nrGridScales):
        cellWidth = 1.0 * min(imgHeight, imgWidth) / (2 ** iter)
        step = cellWidth / 2.0

        for aspectRatio in aspectRatios:
            wStart = 0
            while wStart < imgWidth:
                hStart = 0
                while hStart < imgHeight:
                    if aspectRatio < 1:
                        wEnd = wStart + cellWidth
                        hEnd = hStart + cellWidth / aspectRatio
                    else:
                        wEnd = wStart + cellWidth * aspectRatio
                        hEnd = hStart + cellWidth
                    if wEnd < imgWidth-1 and hEnd < imgHeight-1:
                        rects.append([wStart, hStart, wEnd, hEnd])
                    hStart += step
                wStart += step
    return rects


def filterRois(rects, maxWidth, maxHeight, roi_minNrPixels, roi_maxNrPixels,
               roi_minDim, roi_maxDim, roi_maxAspectRatio):
    filteredRects = []
    filteredRectsSet = set()
    for rect in rects:
        if tuple(rect) in filteredRectsSet: #excluding rectangles with same co-ordinates
            continue

        x, y, x2, y2 = rect
        w = x2 - x
        h = y2 - y
        assert(w>=0 and h>=0)

        # apply filters
        if h == 0 or w == 0 or \
           x2 > maxWidth or y2 > maxHeight or \
           w < roi_minDim or h < roi_minDim or \
           w > roi_maxDim or h > roi_maxDim or \
           w * h < roi_minNrPixels or w * h > roi_maxNrPixels or \
           w / h > roi_maxAspectRatio or h / w > roi_maxAspectRatio:
               continue
        filteredRects.append(rect)
        filteredRectsSet.add(tuple(rect))

    #could combine rectangles using non-maxima surpression or with similar co-ordinates
    #groupedRectangles, weights = cv2.groupRectangles(np.asanyarray(rectsInput, np.float).tolist(), 1, 0.3)
    #groupedRectangles = nms_python(np.asarray(rectsInput, np.float), 0.5)
    assert(len(filteredRects) > 0)
    return filteredRects


def readRois(roiDir, subdir, imgFilename):
    roiPath = roiDir + subdir + "/" + imgFilename[:-4] + ".roi.txt"
    rois = np.loadtxt(roiPath, np.int)
    if len(rois) == 4 and type(rois[0]) == np.int32:  # if only a single ROI in an image
        rois = [rois]
    return rois




####################################
# Generate and parse CNTK files
####################################
def readGtAnnotation(imgPath):
    bboxesPath = imgPath[:-4] + ".bboxes.tsv"
    labelsPath = imgPath[:-4] + ".bboxes.labels.tsv"
    bboxes = np.array(readTable(bboxesPath), np.int32)
    labels = readFile(labelsPath)
    assert (len(bboxes) == len(labels))
    return bboxes, labels


# def cropTransformRoiParams(imgWidth, imgHeight):
#     mindim = np.min([imgWidth, imgHeight])
#     maxdim = np.max([imgWidth, imgHeight])
#     targetw = targeth = mindim
#     cropOffset = 0.5 * (maxdim - mindim)
#     boCropXDim = maxdim == w # compute if crop is horizontal or vertical
#     return targetw, targeth, cropOffset, boCropXDim
#
#
# def cropTransformRoi(rect, cropOffset, boCropXDim, targetImgDim):
#   def offsetMinMax(val, maxVal, crop_offset):
#       val = val - crop_offset
#       val = max(0, val)
#       return min(val, maxVal)
#
#     x, y, x2, y2 = np.asarray(rect)
#     if boCropXDim:
#         x  = offsetMinMax(x,  targetImgDim - 1, cropOffset)
#         x2 = offsetMinMax(x2, targetImgDim - 1, cropOffset)
#     else:
#         y  = offsetMinMax(y,  targetImgDim - 1, cropOffset)
#         y2 = offsetMinMax(y2, targetImgDim - 1, cropOffset)
#     return [x, y, x2, y2]


def getCntkInputPaths(cntkFilesDir, image_set):
    cntkImgsListPath = cntkFilesDir + image_set + '.txt'
    cntkRoiCoordsPath = cntkFilesDir + image_set + '.rois.txt'
    cntkRoiLabelsPath = cntkFilesDir + image_set + '.roilabels.txt'
    cntkNrRoisPath = cntkFilesDir + image_set + '.nrRois.txt'
    return cntkImgsListPath, cntkRoiCoordsPath, cntkRoiLabelsPath, cntkNrRoisPath


def roiTransformPadScaleParams(imgWidth, imgHeight, padWidth, padHeight, boResizeImg = True):
    scale = 1.0
    if boResizeImg:
        assert padWidth == padHeight, "currently only supported equal width/height"
        scale = 1.0 * padWidth / max(imgWidth, imgHeight)
        imgWidth = round(imgWidth * scale)
        imgHeight = round(imgHeight * scale)

    targetw = padWidth
    targeth = padHeight
    w_offset = ((targetw - imgWidth) / 2.)
    h_offset = ((targeth - imgHeight) / 2.)
    if boResizeImg and w_offset > 0 and h_offset > 0:
        print "ERROR: both offsets are > 0:", imgCounter, imgWidth, imgHeight, w_offset, h_offset
        error
    if (w_offset < 0 or h_offset < 0):
        print "ERROR: at least one offset is < 0:", imgWidth, imgHeight, w_offset, h_offset, scale
    return targetw, targeth, w_offset, h_offset, scale


def roiTransformPadScale(rect, w_offset, h_offset, scale = 1.0):
    rect = [int(round(scale * d)) for d in rect]
    rect[0] += w_offset
    rect[1] += h_offset
    rect[2] += w_offset
    rect[3] += h_offset
    return rect


def getCntkRoiCoordsLine(rect, targetw, targeth):
    #convert from absolute to relative co-ordinates
    x, y, x2, y2 = rect
    xrel = float(x) / (1.0 * targetw)
    yrel = float(y) / (1.0 * targeth)
    wrel = float(x2 - x) / (1.0 * targetw)
    hrel = float(y2 - y) / (1.0 * targeth)
    assert xrel <= 1.0, "Error: xrel should be <= 1 but is " + str(xrel)
    assert yrel <= 1.0, "Error: yrel should be <= 1 but is " + str(yrel)
    assert wrel >= 0.0, "Error: wrel should be >= 0 but is " + str(wrel)
    assert hrel >= 0.0, "Error: hrel should be >= 0 but is " + str(hrel)
    return " {} {} {} {}".format(xrel, yrel, wrel, hrel)


def getCntkRoiLabelsLine(overlaps, thres, nrClasses):
    #get one hot encoding
    maxgt = np.argmax(overlaps)
    if overlaps[maxgt] < thres: #set to background label if small overlap with GT
        maxgt = 0
    oneHot = np.zeros((nrClasses), dtype=int)
    oneHot[maxgt] = 1
    oneHotString = " {}".format(" ".join(str(x) for x in oneHot))
    return oneHotString


def cntkPadInputs(currentNrRois, targetNrRois, nrClasses, boxesStr, labelsStr):
    assert currentNrRois <= targetNrRois, "Current number of rois ({}) should be <= target number of rois ({})".format(currentNrRois, targetNrRois)
    while currentNrRois < targetNrRois:
        boxesStr += " 0 0 0 0"
        labelsStr += " 1" + " 0" * (nrClasses - 1)
        currentNrRois += 1
    return boxesStr, labelsStr


def checkCntkOutputFile(cntkImgsListPath, cntkOutputPath, cntkNrRois, outputDim):
    imgPaths = getColumn(readTable(cntkImgsListPath), 1)
    #dnnOutputAccessor = readFileAccessor(cntkOutputPath)
    with open(cntkOutputPath) as fp:
        for imgIndex in range(len(imgPaths)):
            if imgIndex % 100 == 1:
                print "Checking cntk output file, image %d of %d..." % (imgIndex, len(imgPaths))
            for roiIndex in range(cntkNrRois):
                assert (fp.readline() != "")
                #assert (dnnOutputAccessor.next() != "")
        assert (fp.readline() == "") #test if end-of-file is reached


#parse the cntk output file and save the output for each image individually
def parseCntkOutput(cntkImgsListPath, cntkOutputPath, outParsedDir, cntkNrRois, outputDim,
                    saveCompressed = False, skipCheck = False, skip5Mod = None):
    if not skipCheck and skip5Mod == None:
        checkCntkOutputFile(cntkImgsListPath, cntkOutputPath, cntkNrRois, outputDim)

    # parse cntk output and write file for each image
    # always read in data for each image to forward file pointer
    imgPaths = getColumn(readTable(cntkImgsListPath), 1)
    with open(cntkOutputPath) as fp:
        for imgIndex in range(len(imgPaths)):
            line = fp.readline()
            if skip5Mod != None and imgIndex % 5 != skip5Mod:
                print "Skipping image {} (skip5Mod = {})".format(imgIndex, skip5Mod)
                continue
            print "Parsing cntk output file, image %d of %d" % (imgIndex, len(imgPaths))

            # convert to floats
            data = []
            values = np.fromstring(line, dtype=float, sep=" ")
            assert len(values) == cntkNrRois * outputDim, "ERROR: expected dimension of {} but found {}".format(cntkNrRois * outputDim, len(values))
            for i in range(cntkNrRois):
                posStart = i * outputDim
                posEnd = posStart + outputDim
                currValues = values[posStart:posEnd]
                data.append(currValues)

            # save
            data = np.array(data, np.float32)
            outPath = outParsedDir + str(imgIndex) + ".dat"
            if saveCompressed:
                np.savez_compressed(outPath, data)
            else:
                np.savez(outPath, data)
        assert (fp.readline() == "")  # test if end-of-file is reached


#parse the cntk labels file and return the labels
def readCntkRoiLabels(roiLabelsPath, nrRois, roiDim, stopAtImgIndex = None):
    roiLabels = []
    for imgIndex, line in enumerate(readFile(roiLabelsPath)):
        if stopAtImgIndex and imgIndex == stopAtImgIndex:
            break
        roiLabels.append([])
        pos = line.find("|roiLabels ")
        valuesString = line[pos + 10:].strip().split(" ")
        assert (len(valuesString) == nrRois * roiDim)

        for boxIndex in range(nrRois):
            oneHotLabels = [int(s) for s in valuesString[boxIndex*roiDim : (boxIndex+1)*roiDim]]
            assert(sum(oneHotLabels) == 1)
            roiLabels[imgIndex].append(np.argmax(oneHotLabels))
    return roiLabels


#parse the cntk rois file and return the co-ordinates
def readCntkRoiCoordinates(imgPaths, cntkRoiCoordsPath, nrRois, padWidth, padHeight, stopAtImgIndex = None):
    roiCoords = []
    for imgIndex, line in enumerate(readFile(cntkRoiCoordsPath)):
        if stopAtImgIndex and imgIndex == stopAtImgIndex:
            break
        roiCoords.append([])
        pos = line.find("|rois ")
        valuesString = line[pos + 5:].strip().split(" ")
        assert (len(valuesString) == nrRois * 4)

        imgWidth, imgHeight = imWidthHeight(imgPaths[imgIndex])
        for boxIndex in range(nrRois):
            rect = [float(s) for s in valuesString[boxIndex*4 : (boxIndex+1)*4]]
            x,y,w,h = rect
            # convert back from padded-rois-co-ordinates to image co-ordinates
            rect = getAbsoluteROICoordinates([x,y,x+w,y+h], imgWidth, imgHeight, padWidth, padHeight)
            roiCoords[imgIndex].append(rect)
    return roiCoords


#convert roi co-ordinates from CNTK file back to original image co-ordinates
def getAbsoluteROICoordinates(roi, imgWidth, imgHeight, padWidth, padHeight, resizeMethod = 'padScale'):
    if roi == [0,0,0,0]: #if padded roi
        return [0,0,0,0]

    if resizeMethod == "crop":
        minDim = min(imgWidth, imgHeight)
        offsetWidth = 0.5 * abs(imgWidth - imgHeight)
        if (imgWidth >= imgHeight):  # horizontal photo
            rect = [roi[0] * minDim + offsetWidth, roi[1] * minDim, None, None]
        else:
            rect = [roi[0] * minDim, roi[1] * minDim + offsetWidth, None, None]
        rect[2] = rect[0] + roi[2] * minDim
        rect[3] = rect[1] + roi[3] * minDim

    elif resizeMethod == "pad" or resizeMethod == "padScale":
        if resizeMethod == "padScale":
            scale = 1.0 * padWidth / max(imgWidth, imgHeight)
            imgWidthScaled  = int(round(imgWidth * scale))
            imgHeightScaled = int(round(imgHeight * scale))
        else:
            scale = 1.0
            imgWidthScaled = imgWidth
            imgHeightScaled = imgHeight

        w_offset = ((padWidth - imgWidthScaled) / 2.)
        h_offset = ((padHeight - imgHeightScaled) / 2.)
        if resizeMethod == "padScale":
            assert(w_offset == 0 or h_offset == 0)
        rect = [roi[0] * padWidth  - w_offset,
                roi[1] * padHeight - h_offset,
                roi[2] * padWidth  - w_offset,
                roi[3] * padHeight - h_offset]
        rect = [int(round(r / scale)) for r in rect]
    else:
        print "ERROR: Unknown resize method '%s'" % resizeMethod
        error
    assert(min(rect) >=0 and max(rect[0],rect[2]) <= imgWidth and max(rect[1],rect[3]) <= imgHeight)
    return rect




####################################
# Classifier training / scoring
####################################
def getSvmModelPaths(svmDir, experimentName):
    svmWeightsPath   = "{}svmweights_{}.txt".format(svmDir, experimentName)
    svmBiasPath      = "{}svmbias_{}.txt".format(svmDir, experimentName)
    svmFeatScalePath = "{}svmfeature_scale_{}.txt".format(svmDir, experimentName)
    return svmWeightsPath, svmBiasPath, svmFeatScalePath


def loadSvm(svmDir, experimentName):
    svmWeightsPath, svmBiasPath, svmFeatScalePath = getSvmModelPaths(svmDir, experimentName)
    svmWeights   = np.loadtxt(svmWeightsPath, np.float32)
    svmBias      = np.loadtxt(svmBiasPath, np.float32)
    svmFeatScale = np.loadtxt(svmFeatScalePath, np.float32)
    return svmWeights, svmBias, svmFeatScale


def saveSvm(svmDir, experimentName, svmWeights, svmBias, featureScale):
    svmWeightsPath, svmBiasPath, svmFeatScalePath = getSvmModelPaths(svmDir, experimentName)
    np.savetxt(svmWeightsPath, svmWeights)
    np.savetxt(svmBiasPath, svmBias)
    np.savetxt(svmFeatScalePath, featureScale)


def svmPredict(imgIndex, cntkOutputIndividualFilesDir, svmWeights, svmBias, svmFeatScale, roiSize, roiDim, decisionThreshold = 0):
    cntkOutputPath = os.path.join(cntkOutputIndividualFilesDir,  str(imgIndex) + ".dat.npz")
    data = np.load(cntkOutputPath)['arr_0']
    assert(len(data) == roiSize)

    #get prediction for each roi
    labels = []
    maxScores = []
    for roiIndex in range(roiSize):
        feat = data[roiIndex]
        scores = np.dot(svmWeights, feat * 1.0 / svmFeatScale) + svmBias.ravel()
        assert (len(scores) == roiDim)
        maxArg = np.argmax(scores[1:]) + 1
        maxScore = scores[maxArg]
        if maxScore < decisionThreshold:
            maxArg = 0
        labels.append(maxArg)
        maxScores.append(maxScore)
    return labels, maxScores


def nnPredict(imgIndex, cntkParsedOutputDir, roiSize, roiDim, decisionThreshold = None):
    cntkOutputPath = os.path.join(cntkParsedOutputDir,  str(imgIndex) + ".dat.npz")
    data = np.load(cntkOutputPath)['arr_0']
    assert(len(data) == roiSize)

    #get prediction for each roi
    labels = []
    maxScores = []
    for roiIndex in range(roiSize):
        scores = data[roiIndex]
        scores = softmax(scores)
        assert (len(scores) == roiDim)
        maxArg = np.argmax(scores)
        maxScore = scores[maxArg]
        if decisionThreshold and maxScore < decisionThreshold:
            maxArg = 0
        labels.append(maxArg)
        maxScores.append(maxScore)
    return labels, maxScores


def imdbUpdateRoisWithHighGtOverlap(imdb, positivesGtOverlapThreshold):
    addedPosCounter = 0
    existingPosCounter = 0
    for imgIndex in range(imdb.num_images):
        for boxIndex, gtLabel in enumerate(imdb.roidb[imgIndex]['gt_classes']):
            if gtLabel > 0:
                existingPosCounter += 1
            else:
                overlaps = imdb.roidb[imgIndex]['gt_overlaps'][boxIndex, :].toarray()[0]
                maxInd = np.argmax(overlaps)
                maxOverlap = overlaps[maxInd]
                if maxOverlap >= positivesGtOverlapThreshold and maxInd > 0:
                    addedPosCounter += 1
                    imdb.roidb[imgIndex]['gt_classes'][boxIndex] = maxInd
    return existingPosCounter, addedPosCounter




####################################
# Visualize results
####################################
def visualizeResults(imgPath, roiLabels, roiScores, roiRelCoords, padWidth, padHeight, classes,
                     nmsKeepIndices = None, boDrawNegativeRois = True, decisionThreshold = 0.0):
    #read and resize image
    imgWidth, imgHeight = imWidthHeight(imgPath)
    scale = 800.0 / max(imgWidth, imgHeight)
    imgDebug = imresize(imread(imgPath), scale)
    assert(len(roiLabels) == len(roiRelCoords))
    if roiScores:
        assert(len(roiLabels) == len(roiScores))

    #draw multiple times to avoid occlusions
    for iter in range(0,3):
        for roiIndex in range(len(roiRelCoords)):
            label = roiLabels[roiIndex]
            if roiScores:
                score = roiScores[roiIndex]
                if decisionThreshold and score < decisionThreshold:
                    label = 0

            #init drawing parameters
            thickness = 1
            if label == 0:
                color = (255, 0, 0)
            else:
                color = getColorsPalette()[label]
            rect = [int(scale * i) for i in roiRelCoords[roiIndex]]

            #draw in higher iterations only the detections
            if iter == 0 and boDrawNegativeRois:
                drawRectangles(imgDebug, [rect], color=color, thickness=thickness)
            elif iter==1 and label > 0:
                if not nmsKeepIndices or (roiIndex in nmsKeepIndices):
                    thickness = 4
                drawRectangles(imgDebug, [rect], color=color, thickness=thickness)
            elif iter == 2 and label > 0:
                if not nmsKeepIndices or (roiIndex in nmsKeepIndices):
                    font = ImageFont.truetype("arial.ttf", 18)
                    text = classes[label]
                    if roiScores:
                        text += "(" + str(round(score, 2)) + ")"
                    imgDebug = drawText(imgDebug, (rect[0],rect[1]), text, color = (255,255,255), font = font, colorBackground=color)
    return imgDebug


def applyNonMaximaSuppression(nmsThreshold, labels, scores, coords):
    # generate input for nms
    allIndices = []
    nmsRects = [[[]] for _ in xrange(max(labels) + 1)]
    coordsWithScores = np.hstack((coords, np.array([scores]).T))
    for i in range(max(labels) + 1):
        indices = np.where(np.array(labels) == i)[0]
        nmsRects[i][0] = coordsWithScores[indices,:]
        allIndices.append(indices)

    # call nms
    _, nmsKeepIndicesList = apply_nms(nmsRects, nmsThreshold)

    # map back to original roi indices
    nmsKeepIndices = []
    for i in range(max(labels) + 1):
        for keepIndex in nmsKeepIndicesList[i][0]:
            nmsKeepIndices.append(allIndices[i][keepIndex]) # for keepIndex in nmsKeepIndicesList[i][0]]
    assert (len(nmsKeepIndices) == len(set(nmsKeepIndices)))  # check if no roi indices was added >1 times
    return nmsKeepIndices


def apply_nms(all_boxes, thresh, boUsePythonImpl = True):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    nms_keepIndices = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            if boUsePythonImpl:
                keep = nmsPython(dets, thresh)
            else:
                keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
            nms_keepIndices[cls_ind][im_ind] = keep
    return nms_boxes, nms_keepIndices



####################################
# Wrappers for compatibility with
# original fastRCNN code
####################################
class DummyNet(object):
    def __init__(self, dim, num_classes, cntkParsedOutputDir):
        self.name = 'dummyNet'
        self.cntkParsedOutputDir = cntkParsedOutputDir
        self.params = {
            "cls_score": [  EasyDict({'data': np.zeros((num_classes, dim), np.float32) }),
                            EasyDict({'data': np.zeros((num_classes, 1), np.float32) })],
            "trainers" : None,
        }


def im_detect(net, im, boxes, feature_scale=None, bboxIndices=None, boReturnClassifierScore=True, classifier = 'svm'): # trainers=None,
    # Return:
    #     scores (ndarray): R x K array of object class scores (K includes
    #         background as object category 0)
    #     (optional) boxes (ndarray): R x (4*K) array of predicted bounding boxes
    #load cntk output for the given image
    cntkOutputPath = os.path.join(net.cntkParsedOutputDir, str(im) + ".dat.npz")
    cntkOutput = np.load(cntkOutputPath)['arr_0']
    if bboxIndices != None:
        cntkOutput = cntkOutput[bboxIndices, :] # only keep output for certain rois
    else:
        cntkOutput = cntkOutput[:len(boxes), :] # remove zero-padded rois

    #compute scores for each box and each class
    scores = None
    if boReturnClassifierScore:
        if classifier == 'nn':
            scores = softmax2D(cntkOutput)
        elif classifier == 'svm':
            svmBias = net.params['cls_score'][1].data.transpose()
            svmWeights = net.params['cls_score'][0].data.transpose()
            scores = np.dot(cntkOutput * 1.0 / feature_scale, svmWeights) + svmBias
            assert (np.unique(scores[:, 0]) == 0)  # svm always returns 0 for label 0
        else:
            error
    return scores, None, cntkOutput




####################################
# Subset of helper library
# used in the fastRCNN code
####################################
# Typical meaning of variable names -- Computer Vision:
#    pt                     = 2D point (column,row)
#    img                    = image
#    width,height (or w/h)  = image dimensions
#    bbox                   = bbox object (stores: left, top,right,bottom co-ordinates)
#    rect                   = rectangle (order: left, top, right, bottom)
#    angle                  = rotation angle in degree
#    scale                  = image up/downscaling factor

# Typical meaning of variable names -- general:
#    lines,strings = list of strings
#    line,string   = single string
#    xmlString     = string with xml tags
#    table         = 2D row/column matrix implemented using a list of lists
#    row,list1D    = single row in a table, i.e. single 1D-list
#    rowItem       = single item in a row
#    list1D        = list of items, not necessarily strings
#    item          = single item of a list1D
#    slotValue     = e.g. "terminator" in: play <movie> terminator </movie>
#    slotTag       = e.g. "<movie>" or "</movie>" in: play <movie> terminator </movie>
#    slotName      = e.g. "movie" in: play <movie> terminator </movie>
#    slot          = e.g. "<movie> terminator </movie>" in: play <movie> terminator </movie>

import cv2, copy, textwrap
from PIL import Image, ImageFont, ImageDraw
from PIL.ExifTags import TAGS

def makeDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def getFilesInDirectory(directory, postfix = ""):
    fileNames = [s for s in os.listdir(directory) if not os.path.isdir(directory+"/"+s)]
    if not postfix or postfix == "":
        return fileNames
    else:
        return [s for s in fileNames if s.lower().endswith(postfix)]

def readFile(inputFile):
    #reading as binary, to avoid problems with end-of-text characters
    #note that readlines() does not remove the line ending characters
    with open(inputFile,'rb') as f:
        lines = f.readlines()
    return [removeLineEndCharacters(s) for s in lines]

def readTable(inputFile, delimiter='\t', columnsToKeep=None):
    lines = readFile(inputFile);
    if columnsToKeep != None:
        header = lines[0].split(delimiter)
        columnsToKeepIndices = listFindItems(header, columnsToKeep)
    else:
        columnsToKeepIndices = None;
    return splitStrings(lines, delimiter, columnsToKeepIndices)

def getColumn(table, columnIndex):
    column = [];
    for row in table:
        column.append(row[columnIndex])
    return column

def deleteFile(filePath):
    if os.path.exists(filePath):
        os.remove(filePath)

def writeFile(outputFile, lines):
    with open(outputFile,'w') as f:
        for line in lines:
            f.write("%s\n" % line)

def writeTable(outputFile, table):
    lines = tableToList1D(table)
    writeFile(outputFile, lines)

def deleteFile(filePath):
    if os.path.exists(filePath):
        os.remove(filePath)

def deleteAllFilesInDirectory(directory, fileEndswithString, boPromptUser = False):
    if boPromptUser:
        userInput = raw_input('--> INPUT: Press "y" to delete files in directory ' + directory + ": ")
        if not (userInput.lower() == 'y' or userInput.lower() == 'yes'):
            print "User input is %s: exiting now." % userInput
            exit()
    for filename in getFilesInDirectory(directory):
        if fileEndswithString == None or filename.lower().endswith(fileEndswithString):
            deleteFile(directory + "/" + filename)

def removeLineEndCharacters(line):
    if line.endswith('\r\n'):
        return line[:-2]
    elif line.endswith('\n'):
        return line[:-1]
    else:
        return line

def splitString(string, delimiter='\t', columnsToKeepIndices=None):
    if string == None:
        return None
    items = string.split(delimiter)
    if columnsToKeepIndices != None:
        items = getColumns([items], columnsToKeepIndices)
        items = items[0]
    return items;

def splitStrings(strings, delimiter, columnsToKeepIndices=None):
    table = [splitString(string, delimiter, columnsToKeepIndices) for string in strings]
    return table;

def find(list1D, func):
    return [index for (index,item) in enumerate(list1D) if func(item)]

def tableToList1D(table, delimiter='\t'):
    return [delimiter.join([str(s) for s in row]) for row in table]

def sortDictionary(dictionary, sortIndex=0, reverseSort=False):
    return sorted(dictionary.items(), key=lambda x: x[sortIndex], reverse=reverseSort)

def imread(imgPath, boThrowErrorIfExifRotationTagSet = True):
    if not os.path.exists(imgPath):
        "ERROR: image path does not exist."
        error

    rotation = rotationFromExifTag(imgPath)
    if boThrowErrorIfExifRotationTagSet and rotation != 0:
        print "Error: exif roation tag set, image needs to be rotated by %d degrees." % rotation
    img = cv2.imread(imgPath)
    if img is None:
        print "ERROR: cannot load image " + imgPath
        error
    if rotation != 0:
        img = imrotate(img, -90).copy()  # got this error occassionally without copy "TypeError: Layout of the output array img is incompatible with cv::Mat"
    return img

def rotationFromExifTag(imgPath):
    TAGSinverted = {v: k for k, v in TAGS.items()}
    orientationExifId = TAGSinverted['Orientation']
    try:
        imageExifTags = Image.open(imgPath)._getexif()
    except:
        imageExifTags = None

    #rotate the image if orientation exif tag is present
    rotation = 0
    if imageExifTags != None and orientationExifId != None and orientationExifId in imageExifTags:
        orientation = imageExifTags[orientationExifId]
        #print "orientation = " + str(imageExifTags[orientationExifId])
        if orientation == 1 or orientation == 0:
            rotation = 0 #no need to do anything
        elif orientation == 6:
            rotation = -90
        elif orientation == 8:
            rotation = 90
        else:
            print "ERROR: orientation = " + str(orientation) + " not_supported!"
            error
    return rotation

def imwrite(img, imgPath):
    cv2.imwrite(imgPath, img)

def imresize(img, scale, interpolation = cv2.INTER_LINEAR):
    return cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=interpolation)

def imresizeMaxDim(img, maxDim, boUpscale = False, interpolation = cv2.INTER_LINEAR):
    scale = 1.0 * maxDim / max(img.shape[:2])
    if scale < 1  or boUpscale:
        img = imresize(img, scale, interpolation)
    else:
        scale = 1.0
    return img, scale

def imWidth(input):
    return imWidthHeight(input)[0]

def imHeight(input):
    return imWidthHeight(input)[1]

def imWidthHeight(input):
    if type(input) is str or type(input) is unicode:
        width, height = Image.open(input).size #this does not load the full image
    else:
        width =  input.shape[1]
        height = input.shape[0]
    return width,height

def imshow(img, waitDuration=0, maxDim = None, windowName = 'img'):
    if isinstance(img, basestring): #test if 'img' is a string
        img = cv2.imread(img)
    if maxDim is not None:
        scaleVal = 1.0 * maxDim / max(img.shape[:2])
        if scaleVal < 1:
            img = imresize(img, scaleVal)
    cv2.imshow(windowName, img)
    cv2.waitKey(waitDuration)

def drawRectangles(img, rects, color = (0, 255, 0), thickness = 2):
    for rect in rects:
        pt1 = tuple(ToIntegers(rect[0:2]))
        pt2 = tuple(ToIntegers(rect[2:]))
        cv2.rectangle(img, pt1, pt2, color, thickness)

def drawCrossbar(img, pt):
    (x,y) = pt
    cv2.rectangle(img, (0, y), (x, y), (255, 255, 0), 1)
    cv2.rectangle(img, (x, 0), (x, y), (255, 255, 0), 1)
    cv2.rectangle(img, (img.shape[1],y), (x, y), (255, 255, 0), 1)
    cv2.rectangle(img, (x, img.shape[0]), (x, y), (255, 255, 0), 1)

def ptClip(pt, maxWidth, maxHeight):
    pt = list(pt)
    pt[0] = max(pt[0], 0)
    pt[1] = max(pt[1], 0)
    pt[0] = min(pt[0], maxWidth)
    pt[1] = min(pt[1], maxHeight)
    return pt

def drawText(img, pt, text, textWidth=None, color = (255,255,255), colorBackground = None, font = ImageFont.truetype("arial.ttf", 16)):
    pilImg = imconvertCv2Pil(img)
    pilImg = pilDrawText(pilImg,  pt, text, textWidth, color, colorBackground, font)
    return imconvertPil2Cv(pilImg)

def pilDrawText(pilImg, pt, text, textWidth=None, color = (255,255,255), colorBackground = None, font = ImageFont.truetype("arial.ttf", 16)):
    textY = pt[1]
    draw = ImageDraw.Draw(pilImg)
    if textWidth == None:
        lines = [text]
    else:
        lines = textwrap.wrap(text, width=textWidth)
    for line in lines:
        width, height = font.getsize(line)
        if colorBackground != None:
            draw.rectangle((pt[0], pt[1], pt[0] + width, pt[1] + height), fill=tuple(colorBackground[::-1]))
        draw.text(pt, line, fill = tuple(color), font = font)
        textY += height
    return pilImg

def getColorsPalette():
    colors = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255]]
    for i in range(5):
        for dim in range(0,3):
            for s in (0.25, 0.5, 0.75):
                if colors[i][dim] != 0:
                    newColor = copy.deepcopy(colors[i])
                    newColor[dim] = int(round(newColor[dim] * s))
                    colors.append(newColor)
    return colors

def imconvertPil2Cv(pilImg):
    rgb = pilImg.convert('RGB')
    return np.array(rgb).copy()[:, :, ::-1]

def imconvertCv2Pil(img):
    cv2_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_im)

def ToIntegers(list1D):
    return [int(float(x)) for x in list1D]

#TODO: implement numerically stable softmax
def softmax(vec):
    expVec = np.exp(vec)
    #hack
    if max(expVec) == np.inf:
        outVec = np.zeros(len(expVec))
        outVec[expVec == np.inf] = vec[expVec == np.inf]
        outVec = outVec / np.sum(outVec)
    else:
        outVec = expVec / np.sum(expVec)
    return outVec

def softmax2D(w):
    e = np.exp(w)
    dist = e / np.sum(e, axis=1)[:, np.newaxis]
    return dist

def getDictionary(keys, values, boConvertValueToInt = True):
    dictionary = {}
    for key,value in zip(keys, values):
        if (boConvertValueToInt):
            value = int(value)
        dictionary[key] = value
    return dictionary

class Bbox:
    MAX_VALID_DIM = 100000
    left = top = right = bottom = None

    def __init__(self, left, top, right, bottom):
        self.left   = int(round(float(left)))
        self.top    = int(round(float(top)))
        self.right  = int(round(float(right)))
        self.bottom = int(round(float(bottom)))
        self.standardize()

    def __str__(self):
        return ("Bbox object: left = {0}, top = {1}, right = {2}, bottom = {3}".format(self.left, self.top, self.right, self.bottom))

    def __repr__(self):
        return str(self)

    def rect(self):
        return [self.left, self.top, self.right, self.bottom]

    def max(self):
        return max([self.left, self.top, self.right, self.bottom])

    def min(self):
        return min([self.left, self.top, self.right, self.bottom])

    def width(self):
        width  = self.right - self.left + 1
        assert(width>=0)
        return width

    def height(self):
        height = self.bottom - self.top + 1
        assert(height>=0)
        return height

    def surfaceArea(self):
        return self.width() * self.height()

    def getOverlapBbox(self, bbox):
        left1, top1, right1, bottom1 = self.rect()
        left2, top2, right2, bottom2 = bbox.rect()
        overlapLeft = max(left1, left2)
        overlapTop = max(top1, top2)
        overlapRight = min(right1, right2)
        overlapBottom = min(bottom1, bottom2)
        if (overlapLeft>overlapRight) or (overlapTop>overlapBottom):
            return None
        else:
            return Bbox(overlapLeft, overlapTop, overlapRight, overlapBottom)

    def standardize(self): #NOTE: every setter method should call standardize
        leftNew   = min(self.left, self.right)
        topNew    = min(self.top, self.bottom)
        rightNew  = max(self.left, self.right)
        bottomNew = max(self.top, self.bottom)
        self.left = leftNew
        self.top = topNew
        self.right = rightNew
        self.bottom = bottomNew

    def crop(self, maxWidth, maxHeight):
        leftNew   = min(max(self.left,   0), maxWidth)
        topNew    = min(max(self.top,    0), maxHeight)
        rightNew  = min(max(self.right,  0), maxWidth)
        bottomNew = min(max(self.bottom, 0), maxHeight)
        return Bbox(leftNew, topNew, rightNew, bottomNew)

    def isValid(self):
        if self.left>=self.right or self.top>=self.bottom:
            return False
        if min(self.rect()) < -self.MAX_VALID_DIM or max(self.rect()) > self.MAX_VALID_DIM:
            return False
        return True

def getEnclosingBbox(pts):
    left = top = float('inf')
    right = bottom = float('-inf')
    for pt in pts:
        left   = min(left,   pt[0])
        top    = min(top,    pt[1])
        right  = max(right,  pt[0])
        bottom = max(bottom, pt[1])
    return Bbox(left, top, right, bottom)

def bboxComputeOverlapVoc(bbox1, bbox2):
    surfaceRect1 = bbox1.surfaceArea()
    surfaceRect2 = bbox2.surfaceArea()
    overlapBbox = bbox1.getOverlapBbox(bbox2)
    if overlapBbox == None:
        return 0
    else:
        surfaceOverlap = overlapBbox.surfaceArea()
        overlap = max(0, 1.0 * surfaceOverlap / (surfaceRect1 + surfaceRect2 - surfaceOverlap))
        assert (overlap >= 0 and overlap <= 1)
        return overlap

def computeAveragePrecision(recalls, precisions, use_07_metric=False):
    """ ap = voc_ap(recalls, precisions, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrecalls = np.concatenate(([0.], recalls, [1.]))
        mprecisions = np.concatenate(([0.], precisions, [0.]))

        # compute the precision envelope
        for i in range(mprecisions.size - 1, 0, -1):
            mprecisions[i - 1] = np.maximum(mprecisions[i - 1], mprecisions[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrecalls[1:] != mrecalls[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrecalls[i + 1] - mrecalls[i]) * mprecisions[i + 1])
    return ap