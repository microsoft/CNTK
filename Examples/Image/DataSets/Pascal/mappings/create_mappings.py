# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
#
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import sys, os
import numpy as np
import xml.etree.ElementTree
from xml.etree import ElementTree
from enum import Enum
from PIL import Image

# (ctrx, ctry, w, h) in relative coords (e.g. for Yolo)
use_relative_coords_ctr_wh = False
# else: top left and bottom right corner are used (i.e. xmin, ymin, xmax, ymax) in absolute coords

use_pad_scale = False
pad_width = 850
pad_height = 850

pascal_voc2007_jpgimg_rel_path = "../VOCdevkit/VOC2007/JPEGImages/"
pascal_voc2007_imgsets_rel_path = "../VOCdevkit/VOC2007/ImageSets/Main/"
pascal_voc2007_annotations_rel_path = "../VOCdevkit/VOC2007/Annotations/"

abs_path = os.path.dirname(os.path.abspath(__file__))
cls_file_path = os.path.join(abs_path, "class_map.txt")

classes = ('__background__',  # always index 0
           'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
class_dict = {k: v for v, k in enumerate(classes)}

def format_roi(cls_index, xmin, ymin, xmax, ymax, img_file_path):
    posx = xmin
    posy = ymin
    width = (xmax - xmin)
    height = (ymax - ymin)

    if use_pad_scale or use_relative_coords_ctr_wh:
        img_width, img_height = Image.open(img_file_path).size

        if use_pad_scale:
            scale_x = (1.0 * pad_width) / img_width
            scale_y = (1.0 * pad_height) / img_height

            min_scale = min(scale_x, scale_y)
            if round(img_width * min_scale) != pad_width and round(img_height * min_scale) != pad_height:
                import pdb; pdb.set_trace()

            new_width = round(img_width * min_scale)
            new_height = round(img_height * min_scale)
            assert(new_width == pad_width or new_height == pad_height)
            assert(new_width <= pad_width and new_height <= pad_height)

            offset_x = (pad_width - new_width) / 2
            offset_y = (pad_height - new_height) / 2

            width = round(width * min_scale)
            height = round(height * min_scale)
            posx = round(posx * min_scale + offset_x)
            posy = round(posy * min_scale + offset_y)

            norm_width = pad_width
            norm_height = pad_height
        else:
            norm_width = img_width
            norm_height = img_height

        if use_relative_coords_ctr_wh:
            ctrx = xmin + width / 2
            ctry = ymin + height / 2

            width = float(width) / norm_width
            height = float(height) / norm_height
            ctrx = float (ctrx) / norm_width
            ctry = float(ctry) / norm_height

    if use_relative_coords_ctr_wh:
        return "{:.4f} {:.4f} {:.4f} {:.4f} {} ".format(ctrx, ctry, width, height, cls_index)
    else:
        posx2 = posx + width
        posy2 = posy + height
        return "{} {} {} {} {} ".format(int(posx), int(posy), int(posx2), int(posy2), cls_index)

def create_mappings(train, skip_difficult):
    file_prefix = "trainval" if train else "test"
    img_map_input = "../VOCdevkit/VOC2007/ImageSets/Main/{}.txt".format(file_prefix)
    img_map_output = "{}2007.txt".format(file_prefix)
    roi_map_output = "{}2007_rois_{}_{}{}.txt".format(
        file_prefix,
        "rel-ctr-wh" if use_relative_coords_ctr_wh else "abs-xyxy",
        "pad" if use_pad_scale else "noPad",
        "_skipDif" if skip_difficult else "")
    size_map_output = "{}_size_file2007.txt".format(file_prefix)

    in_map_file_path = os.path.join(abs_path, img_map_input)
    out_map_file_path = os.path.join(abs_path, img_map_output)
    roi_file_path = os.path.join(abs_path, roi_map_output)
    size_file_path = os.path.join(abs_path, size_map_output)
    class_map_file_path = os.path.join(abs_path, "class_map.txt")

    # write class map file
    class_list = [None]*len(class_dict)
    for k in class_dict:
        class_list[class_dict[k]]=k
    with open(class_map_file_path, 'w') as class_map_file:
        for i in range(len(class_list)):
            class_map_file.write("{}\t{}\n".format(class_list[i], i))

    # read input file
    with open(in_map_file_path) as input_file:
        input_lines = input_file.readlines()

    counter = 0
    with open(out_map_file_path, 'w') as img_file:
        with open(roi_file_path, 'w') as roi_file:
            with open(size_file_path, 'w') as size_file:
                for in_line in input_lines:
                    img_number = in_line.strip()
                    img_file_path = "{}{}.jpg".format(pascal_voc2007_jpgimg_rel_path, img_number)
                    img_line = "{}\t{}\t0\n".format(counter, img_file_path)
                    img_file.write(img_line)

                    annotation_file = os.path.join(pascal_voc2007_annotations_rel_path, "{}.xml".format(img_number))
                    annotations = ElementTree.parse(annotation_file).getroot()

                    roi_line = "{} |roiAndLabel ".format(counter)
                    for obj in annotations.findall('object'):
                        if skip_difficult:
                            difficult = int(obj.findall('difficult')[0].text)
                            if difficult == 1:
                                continue

                        cls = obj.findall('name')[0].text
                        cls_index = class_dict[cls]

                        bbox = obj.findall('bndbox')[0]
                        # subtracting 1 since matlab indexing is 1-based
                        xmin = int(bbox.findall('xmin')[0].text) - 1
                        ymin = int(bbox.findall('ymin')[0].text) - 1
                        xmax = int(bbox.findall('xmax')[0].text) - 1
                        ymax = int(bbox.findall('ymax')[0].text) - 1

                        assert xmin >= 0 and ymin >= 0 and xmax >= 0 and ymax >=0

                        roi_line += format_roi(cls_index, xmin, ymin, xmax, ymax, img_file_path)

                    roi_file.write(roi_line + "\n")

                    size_line = "{} |size".format(counter)
                    with Image.open(img_file_path) as img:
                        width, height = img.size
                    size_line += " {} {}\n".format(width, height)
                    size_file.write(size_line)

                    counter += 1
                    if counter % 500 == 0:
                        print("Processed {} images".format(counter))

    with open(cls_file_path, 'w') as cls_file:
        for cls in classes:
            cls_file.write("{}\t{}\n".format(cls, class_dict[cls]))

if __name__ == '__main__':
    create_mappings(True, skip_difficult=True)
    create_mappings(False, skip_difficult=True)
    create_mappings(True, skip_difficult=False)
    create_mappings(False, skip_difficult=False)
