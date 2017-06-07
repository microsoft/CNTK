# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import os.path
import tarfile

from xml.etree import ElementTree
from PIL import Image

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

use_center_of_bbox = True  # top left corner (i.e. xmin, ymin) will be used if set to False
use_relative_coords = True  # absolute pixel coordinates will be used if set to False

pascal_voc2007_jpgimg_rel_path = os.path.join("VOCdevkit", "VOC2007", "JPEGImages")
pascal_voc2007_annotations_rel_path = os.path.join("VOCdevkit", "VOC2007", "Annotations")

img_map_output = os.path.join("mappings", "trainval2007.txt")
roi_map_output = os.path.join("mappings",
                              "trainval2007_rois_{}_{}.txt".format("center" if use_center_of_bbox else "topleft",
                                                                   "rel" if use_relative_coords else "abs"))


def download_and_untar(url, path, filename, filesize):
    if not os.path.exists(filename):
        print('Downloading ' + filesize + ' from ' + url + ', may take a while...')
        try:
            urlretrieve(url, filename)
        except (urllib.ContentTooShortError, IOError) as e:
            print("Error downloading file: " + str(e))
            os.remove(filename)
            quit()
    else:
        print('Found ' + filename)
    try:
        print('Extracting ' + filename + '...')
        with tarfile.open(filename) as tar:
            tar.extractall(path=path)
        print('Done.')
    finally:
        os.remove(filename)
    return


def download_model(model_file, model_url):
    if not os.path.exists(model_file):
        print('Downloading model from ' + model_url + ', may take a while...')
        urlretrieve(model_url, model_file)
        print('Saved model as ' + model_file)
    else:
        print('CNTK model already available at ' + model_file)


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
        posx = float(posx) / img_width
        posy = float(posy) / img_height
        return "{:.4f} {:.4f} {:.4f} {:.4f} {} ".format(posx, posy, width, height, cls_index)
    else:
        return "{} {} {} {} {} ".format(posx, posy, width, height, cls_index)


if __name__ == "__main__":
    base_folder = os.path.dirname(os.path.abspath(__file__))

    # Download Pascal Data
    pascal_directory = os.path.join(base_folder, "..", "..", "DataSets", "Pascal")
    print("Downloading Pascal VOC data to: " + pascal_directory)

    pascal2007_dir = os.path.join(pascal_directory, "VOCdevkit", "VOC2007")
    if not os.path.exists(pascal2007_dir):
        download_and_untar(
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
            pascal_directory,
            os.path.join(pascal_directory, "VOCtrainval_06-Nov-2007.tar"),
            "450MB")
        download_and_untar(
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
            pascal_directory,
            os.path.join(pascal_directory, "VOCtest_06-Nov-2007.tar"),
            "430MB")
    else:
        print(pascal2007_dir + ' data already available.')

    if False:  # maybe for future use!
        pascal2012_dir = os.path.join(pascal_directory, "VOCdevkit", "VOC2012")
        if not os.path.exists(pascal2012_dir):
            download_and_untar(
                "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
                pascal_directory,
                os.path.join(pascal_directory, "VOCtrainval_11-May-2012.tar"),
                "2GB")
        else:
            print(pascal2012_dir + ' data already available.')

        selsearch_directory = os.path.join(pascal_directory, "selective_search_data")
        if not os.path.exists(selsearch_directory):
            os.makedirs(selsearch_directory)
            download_and_untar(
                "http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/selective_search_data.tgz",
                pascal_directory,
                os.path.join(pascal_directory, "selective_search_data.tgz"),
                "460MB")
        else:
            print(selsearch_directory + ' data already available.')
    print("Finished downloading data!\n")

    # get Model!
    print("Downloading Models:")
    print("ResNet101_ImageNet")
    model_file = os.path.join(base_folder, "..", "..", "PretrainedModels", "ResNet101_ImageNet.model")
    download_model(model_file, "https://www.cntk.ai/Models/Caffe_Converted/ResNet101_ImageNet.model")
    print("Finished downloading models!\n")

    # create mappings!
    in_map_file_path = os.path.join(pascal_directory, "VOCdevkit", "VOC2007", "ImageSets", "Main", "trainval.txt")
    out_map_file_path = os.path.join(pascal_directory, img_map_output)
    roi_file_path = os.path.join(pascal_directory, roi_map_output)
    cls_file_path = os.path.join(pascal_directory, "class_map.txt")

    classes = ('__background__',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    class_dict = {k: v for v, k in enumerate(classes)}

    mappings_dir = os.path.join(pascal_directory, "mappings")
    if not os.path.exists(mappings_dir):
        os.makedirs(mappings_dir)

    if not os.path.exists(out_map_file_path):
        with open(in_map_file_path) as input_file:
            input_lines = input_file.readlines()

        counter = 0
        with open(out_map_file_path, 'w') as img_file:
            with open(roi_file_path, 'w') as roi_file:
                for in_line in input_lines:
                    img_number = in_line.strip()
                    img_file_path = "{}{}.jpg".format(
                        os.path.join(pascal_directory, pascal_voc2007_jpgimg_rel_path, ""), img_number)
                    img_line = "{}\t{}\t0\n".format(counter, img_file_path)
                    img_file.write(img_line)

                    annotation_file = os.path.join(pascal_directory, pascal_voc2007_annotations_rel_path,
                                                   "{}.xml".format(img_number))
                    annotations = ElementTree.parse(annotation_file).getroot()

                    roi_line = "{} |rois ".format(counter)
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

    if not os.path.exists(cls_file_path):
        with open(cls_file_path, 'w') as cls_file:
            for cls in classes:
                cls_file.write("{}\t{}\n".format(cls, class_dict[cls]))
