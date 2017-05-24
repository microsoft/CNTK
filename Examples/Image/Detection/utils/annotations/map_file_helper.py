# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import cv2
import numpy as np
import os

# TODO: generate class list from annotations
classes = ('__background__',  # always index 0
                'avocado', 'orange', 'butter', 'champagne', 'eggBox', 'gerkin', 'joghurt', 'ketchup',
                'orangeJuice', 'onion', 'pepper', 'tomato', 'water', 'milk', 'tabasco', 'mustard')
class_dict = {k: v for v, k in enumerate(classes)}

def _getFilesInDirectory(directory, postfix = ""):
    fileNames = [s for s in os.listdir(directory) if not os.path.isdir(os.path.join(directory, s))]
    if not postfix or postfix == "":
        return fileNames
    else:
        return [s for s in fileNames if s.lower().endswith(postfix)]

def _get_image_paths(img_dir, training_set):
    if training_set:
        subDirs = ['positive', 'negative']
    else:
        subDirs = ['testImages']

    image_paths = []
    for subdir in subDirs:
        sub_dir_path = os.path.join(img_dir, subdir)
        imgFilenames = _getFilesInDirectory(sub_dir_path, ".jpg")
        for img in imgFilenames:
            image_paths.append("{}/{}".format(subdir, img))

    return image_paths

def _removeLineEndCharacters(line):
    if line.endswith(b'\r\n'):
        return line[:-2]
    elif line.endswith(b'\n'):
        return line[:-1]
    else:
        return line

def _load_annotation(imgPath):
    bboxesPaths = imgPath[:-4] + ".bboxes.tsv"
    labelsPaths = imgPath[:-4] + ".bboxes.labels.tsv"
    # if no ground truth annotations are available, return None
    if not os.path.exists(bboxesPaths) or not os.path.exists(labelsPaths):
        return None
    bboxes = np.loadtxt(bboxesPaths, np.float32)

    # in case there's only one annotation and numpy read the array as single array,
    # we need to make sure the input is treated as a multi dimensional array instead of a list/ 1D array
    if len(bboxes.shape) == 1:
        bboxes = np.array([bboxes])

    with open(labelsPaths, 'rb') as f:
        lines = f.readlines()
    labels = [_removeLineEndCharacters(s) for s in lines]

    #remove boxes marked as 'undecided' or 'exclude' ?
    #indicesToKeep = find(labels, lambda x: x!='EXCLUDE' and x!='UNDECIDED')
    #bboxes = [bboxes[i] for i in indicesToKeep]
    #labels = [labels[i] for i in indicesToKeep]
    label_idxs = np.asarray([class_dict[l.decode('utf-8')] for l in labels])
    label_idxs.shape = label_idxs.shape + (1,)
    annotations = np.hstack((bboxes, label_idxs))

    return annotations

def create_map_files(data_folder, training_set):
    # get relative paths for map files
    img_file_paths = _get_image_paths(data_folder, training_set)

    out_map_file_path = os.path.join(data_folder, "{}_img_file.txt".format("train" if training_set else "test"))
    roi_file_path = os.path.join(data_folder, "{}_roi_file.txt".format("train" if training_set else "test"))

    counter = 0
    with open(out_map_file_path, 'w') as img_file:
        with open(roi_file_path, 'w') as roi_file:
            for img_path in img_file_paths:
                abs_img_path = os.path.join(data_folder, img_path)
                gt_annotations = _load_annotation(abs_img_path)
                if gt_annotations is None:
                    continue

                img_line = "{}\t{}\t0\n".format(counter, img_path)
                img_file.write(img_line)

                roi_line = "{} |roiAndLabel".format(counter)
                for val in gt_annotations.flatten():
                    roi_line += " {}".format(val)

                roi_file.write(roi_line + "\n")
                counter += 1
                if counter % 500 == 0:
                    print("Processed {} images".format(counter))

if __name__ == '__main__':
    create_map_files(r"C:\src\CNTK\Examples\Image\DataSets\Grocery", training_set=True)
    create_map_files(r"C:\src\CNTK\Examples\Image\DataSets\Grocery", training_set=False)
