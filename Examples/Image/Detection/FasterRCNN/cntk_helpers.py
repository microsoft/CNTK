from __future__ import print_function
from builtins import str
import pdb, sys, os, time
import numpy as np
from easydict import EasyDict
from builtins import range
import cv2, copy, textwrap
from PIL import Image, ImageFont, ImageDraw
from PIL.ExifTags import TAGS

available_font = "arial.ttf"
try:
    dummy = ImageFont.truetype(available_font, 16)
except:
    available_font = "FreeMono.ttf"

####################################
# Visualize results
####################################
def visualizeResultsFaster(imgPath, roiLabels, roiScores, roiRelCoords, padWidth, padHeight, classes,
                     nmsKeepIndices = None, boDrawNegativeRois = True, decisionThreshold = 0.0):
    # read and resize image
    imgWidth, imgHeight = imWidthHeight(imgPath)
    scale = 800.0 / max(imgWidth, imgHeight)
    imgHeight = int(imgHeight * scale)
    imgWidth = int(imgWidth * scale)
    if imgWidth > imgHeight:
        h_border = 0
        v_border = int((imgWidth - imgHeight)/2)
    else:
        h_border = int((imgHeight - imgWidth)/2)
        v_border = 0

    PAD_COLOR = [114, 114, 114]
    cv_img = cv2.imread(imgPath)
    rgb_img = cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_img, (imgWidth, imgHeight), interpolation=cv2.INTER_NEAREST)
    imgDebug = cv2.copyMakeBorder(resized,v_border,v_border,h_border,h_border,cv2.BORDER_CONSTANT,value=PAD_COLOR)
    rect_scale = 800 / padWidth

    assert(len(roiLabels) == len(roiRelCoords))
    if roiScores:
        assert(len(roiLabels) == len(roiScores))

    # draw multiple times to avoid occlusions
    for iter in range(0,3):
        for roiIndex in range(len(roiRelCoords)):
            label = roiLabels[roiIndex]
            if roiScores:
                score = roiScores[roiIndex]
                if decisionThreshold and score < decisionThreshold:
                    label = 0

            # init drawing parameters
            thickness = 1
            if label == 0:
                color = (255, 0, 0)
            else:
                color = getColorsPalette()[label]

            rect = [(rect_scale * i) for i in roiRelCoords[roiIndex]]
            rect[0] = int(max(0, min(padWidth, rect[0])))
            rect[1] = int(max(0, min(padHeight, rect[1])))
            rect[2] = int(max(0, min(padWidth, rect[2])))
            rect[3] = int(max(0, min(padHeight, rect[3])))

            # draw in higher iterations only the detections
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


def imread(imgPath, boThrowErrorIfExifRotationTagSet = True):
    if not os.path.exists(imgPath):
        print("ERROR: image path does not exist.")
        error

    rotation = rotationFromExifTag(imgPath)
    if boThrowErrorIfExifRotationTagSet and rotation != 0:
        print ("Error: exif roation tag set, image needs to be rotated by %d degrees." % rotation)
    img = cv2.imread(imgPath)
    if img is None:
        print ("ERROR: cannot load image " + imgPath)
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

    # rotate the image if orientation exif tag is present
    rotation = 0
    if imageExifTags != None and orientationExifId != None and orientationExifId in imageExifTags:
        orientation = imageExifTags[orientationExifId]
        # print ("orientation = " + str(imageExifTags[orientationExifId]))
        if orientation == 1 or orientation == 0:
            rotation = 0 # no need to do anything
        elif orientation == 6:
            rotation = -90
        elif orientation == 8:
            rotation = 90
        else:
            print ("ERROR: orientation = " + str(orientation) + " not_supported!")
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
    width, height = Image.open(input).size #this does not load the full image
    return width,height

def imArrayWidth(input):
    return imArrayWidthHeight(input)[0]

def imArrayHeight(input):
    return imArrayWidthHeight(input)[1]
    
def imArrayWidthHeight(input):
    width =  input.shape[1]
    height = input.shape[0]
    return width,height
 
def imshow(img, waitDuration=0, maxDim = None, windowName = 'img'):
    if isinstance(img, str): #test if 'img' is a string
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
        try:
            cv2.rectangle(img, pt1, pt2, color, thickness)
        except:
            import pdb; pdb.set_trace()
            print("Unexpected error:", sys.exc_info()[0])

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

def softmax(vec):
    expVec = np.exp(vec)
    # TODO: check numerical stability
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


