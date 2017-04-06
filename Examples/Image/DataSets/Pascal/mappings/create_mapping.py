import sys, os
import numpy as np
import xml.etree.ElementTree
from xml.etree import ElementTree
from enum import Enum
from PIL import Image

use_center_of_bbox = False      # top left corner (i.e. xmin, ymin) will be used if set to False
use_relative_coords = True      # absolute pixel coordinates will be used if set to False

pascal_voc2007_jpgimg_rel_path = "../VOCdevkit/VOC2007/JPEGImages/"
pascal_voc2007_imgsets_rel_path = "../VOCdevkit/VOC2007/ImageSets/Main/"
pascal_voc2007_annotations_rel_path = "../VOCdevkit/VOC2007/Annotations/"

train = False
if train:
    img_map_input = "../VOCdevkit/VOC2007/ImageSets/Main/trainval.txt"
    img_map_output = "trainval2007.txt"
    roi_map_output = "trainval2007_rois_{}_wh_{}.txt".format("center" if use_center_of_bbox else "topleft", "rel" if use_relative_coords else "abs")
else:
    img_map_input = "../VOCdevkit/VOC2007/ImageSets/Main/test.txt"
    img_map_output = "testval2007.txt"
    roi_map_output = "testval2007_rois_{}_wh_{}.txt".format("center" if use_center_of_bbox else "topleft", "rel" if use_relative_coords else "abs")

abs_path = os.path.dirname(os.path.abspath(__file__))
in_map_file_path = os.path.join(abs_path, img_map_input)
out_map_file_path = os.path.join(abs_path, img_map_output)
roi_file_path = os.path.join(abs_path, roi_map_output)
cls_file_path = os.path.join(abs_path, "class_map.txt")

classes = ('__background__',  # always index 0
           'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
class_dict = {k: v for v, k in enumerate(classes)}

def format_roi(cls_index, xmin, ymin, xmax, ymax, img_file_path):
    if use_center_of_bbox:
        width = (xmax - xmin)
        height = (ymax - ymin)
        posx = xmin + width / 2
        posy = ymin + height / 2 
    else:
        width = (xmax - xmin)
        height = (ymax - ymin)
        posx = xmin
        posy = ymin

    if use_relative_coords:
        img_width, img_height = Image.open(img_file_path).size
        width = float(width) / img_width
        height = float(height) / img_height
        posx = float (posx) / img_width
        posy = float(posy) / img_height
        return "{:.4f} {:.4f} {:.4f} {:.4f} {} ".format(posx, posy, width, height, cls_index)
    else:
        return "{} {} {} {} {} ".format(posx, posy, width, height, cls_index)

if __name__ == '__main__':
    with open(in_map_file_path) as input_file:
        input_lines = input_file.readlines()

    counter = 0
    with open(out_map_file_path, 'w') as img_file:
        with open(roi_file_path, 'w') as roi_file:
            for in_line in input_lines:
                img_number = in_line.strip()
                img_file_path = "{}{}.jpg".format(pascal_voc2007_jpgimg_rel_path, img_number)
                img_line = "{}\t{}\t0\n".format(counter, img_file_path)
                img_file.write(img_line)

                annotation_file = os.path.join(pascal_voc2007_annotations_rel_path, "{}.xml".format(img_number))
                annotations = ElementTree.parse(annotation_file).getroot()

                roi_line = "{} |roiAndLabel ".format(counter)
                for obj in annotations.findall('object'):
                    cls = obj.findall('name')[0].text
                    cls_index = class_dict[cls]

                    bbox = obj.findall('bndbox')[0]
                    xmin = int(bbox.findall('xmin')[0].text)
                    ymin = int(bbox.findall('ymin')[0].text)
                    xmax = int(bbox.findall('xmax')[0].text)
                    ymax = int(bbox.findall('ymax')[0].text)

                    roi_line += format_roi(cls_index, xmin, ymin, xmax, ymax, img_file_path)

                roi_file.write(roi_line + "\n")
                counter += 1
                if counter % 500 == 0:
                    print("Processed {} images".format(counter))

    with open(cls_file_path, 'w') as cls_file:
        for cls in classes:
            cls_file.write("{}\t{}\n".format(cls, class_dict[cls]))
