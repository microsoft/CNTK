import os
import xml.etree.ElementTree as ET
import csv

filepath = "C:/Your/Folder/Labelme/Files/" # set path of Labelme XML Files here include slash at end of path

for filename in os.listdir(filepath):
    try:
        file = filepath + filename

        tree = ET.parse(file)
        root = tree.getroot()

        outputpath = filepath + "Parsed/"

        if not os.path.exists(outputpath):
            os.makedirs(outputpath)

        imagename = os.path.splitext(filename)[0]

        ## create output files  
        outputFile_label = outputpath + imagename + ".bboxes.labels.tsv"
        outputFile_ROI = outputpath + imagename + ".bboxes.tsv"

        labelFile = open(outputFile_label, 'w')
        ROIFile = open(outputFile_ROI, 'w')

        # loop through to get objects
        for child in root:
            if str(child.tag) == 'object':

                label = ""
                xlist = []
                ylist = []

                # loop through to get name and BBox values from object
                for child in child:
                    if str(child.tag) == 'name':
                        label = child.text
                    if str(child.tag) == 'polygon' or str(child.tag) == 'segm':
                        for child in child:
                             if str(child.tag) == 'box' or str(child.tag) == 'pt':
                                for child in child:
                                    if str(child.tag) == 'xmin' or str(child.tag) == 'xmax' or str(child.tag) == 'x':
                                            xlist.append(int(child.text))
                                    if str(child.tag) == 'ymin' or str(child.tag) == 'ymax' or str(child.tag) == 'y':
                                            ylist.append(int(child.text))

                xmin = min(xlist)
                xmax = max(xlist)

                ymin = min(ylist)
                ymax = max(ylist)

                # output object roi based on cntk format of xmin ymin xmax ymax
                obj_ROI = str(xmin) + "\t" + str(ymin) + "\t" +str(xmax) + "\t" + str(ymax)

                labelFile.write(label  + '\n')
                ROIFile.write(obj_ROI  + '\n')

        labelFile.close()
        ROIFile.close()

    except Exception:
        pass

print("Done")
