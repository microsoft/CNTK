# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
try:
    # for Python2
    from Tkinter import *
except ImportError:
    # for Python3
    from tkinter import *
from PIL import ImageTk
from cntk_helpers import *


####################################
# Parameters
####################################
imgDir = "C:/Users/chazhang/Desktop/newImgs/"
classes = ("avocado", "orange", "butter", "champagne", "cheese", "eggBox", "gerkin", "joghurt", "ketchup",
           "orangeJuice", "onion", "pepper", "sausage", "tomato", "water", "apple", "milk",
           "tabasco", "soySauce", "mustard", "beer")

#no need to change these
drawingImgSize = 1000
boxWidth = 10
boxHeight = 2


####################################
# Main
####################################
# define callback function for tk button
def buttonPressedCallback(s):
    global global_lastButtonPressed
    global_lastButtonPressed = s

# create UI
objectNames = np.sort(classes).tolist()
objectNames += ["UNDECIDED", "EXCLUDE"]
tk = Tk()
w = Canvas(tk, width=len(objectNames) * boxWidth, height=len(objectNames) * boxHeight, bd = boxWidth, bg = 'white')
w.grid(row = len(objectNames), column = 0, columnspan = 2)
for objectIndex,objectName in enumerate(objectNames):
    b = Button(width=boxWidth, height=boxHeight, text=objectName, command=lambda s = objectName: buttonPressedCallback(s))
    b.grid(row = objectIndex, column = 0)

# loop over all images
imgFilenames = getFilesInDirectory(imgDir, ".jpg")
for imgIndex, imgFilename in enumerate(imgFilenames):
    print (imgIndex, imgFilename)
    labelsPath = os.path.join(imgDir, imgFilename[:-4] + ".bboxes.labels.tsv")
    if os.path.exists(labelsPath):
        print ("Skipping image {:3} ({}) since annotation file already exists: {}".format(imgIndex, imgFilename, labelsPath))
        continue

    # load image and ground truth rectangles
    img = imread(os.path.join(imgDir,imgFilename))
    rectsPath = os.path.join(imgDir, imgFilename[:-4] + ".bboxes.tsv")
    rects = [ToIntegers(rect) for rect in readTable(rectsPath)]

    # annotate each rectangle in turn
    labels = []
    for rectIndex,rect in enumerate(rects):
        imgCopy = img.copy()
        drawRectangles(imgCopy, [rect], thickness = 15)

        # draw image in tk window
        imgTk, _ = imresizeMaxDim(imgCopy, drawingImgSize, boUpscale = True)
        imgTk = ImageTk.PhotoImage(imconvertCv2Pil(imgTk))
        label = Label(tk, image=imgTk)
        label.grid(row=0, column=1, rowspan=drawingImgSize)
        tk.update_idletasks()
        tk.update()

        # busy-wait until button pressed
        global_lastButtonPressed = None
        while not global_lastButtonPressed:
            tk.update_idletasks()
            tk.update()

        # store result
        print ("Button pressed = ", global_lastButtonPressed)
        labels.append(global_lastButtonPressed)

    writeFile(labelsPath, labels)
tk.destroy()
print ("DONE.")