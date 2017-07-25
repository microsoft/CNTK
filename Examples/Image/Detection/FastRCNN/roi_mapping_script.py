# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
#
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import PARAMETERS as params

p = params.get_parameters_for_dataset()
a1_path = p.cntkFilesDir
img_file_a1 = os.path.join(a1_path, "test.txt")

roi_file_a1 = os.path.join(a1_path, "test.rois.txt")

roi_file_dataset_destination = os.path.join(a1_path, "test.rois.ds.txt")

dataset_path = p.imgDir
if p.datasetName != "pascalVoc":
    img_file_dataset = os.path.join(dataset_path, "test_img_file.txt")
else:
    img_file_dataset = os.path.join(dataset_path, "mappings", "test2007.txt")

print("Remapping rois to datasets order...")

# read a1 mapping
input_file_a1_img = open(img_file_a1)
input_map = {}

for line in input_file_a1_img.readlines():
    if line != "":
        pieces = line.split()
        if os.sys.platform == 'win32':
            input_map[int(pieces[0])] = os.path.normpath(os.path.abspath(pieces[1])).lower()
        else:
            input_map[int(pieces[0])] = os.path.normpath(os.path.abspath(pieces[1]))

input_file_a1_img.close()

# read dataset mapping
input_file_dataset_img = open(img_file_dataset)

output_map = {}
for line in input_file_dataset_img.readlines():
    if line != "":
        pieces = line.split()
        if os.sys.platform=='win32':
            output_map[os.path.normpath(os.path.abspath(pieces[1])).lower()] = int(pieces[0])
        else:
            output_map[os.path.normpath(os.path.abspath(pieces[1]))] = int(pieces[0])

input_file_dataset_img.close()
#import ipdb;ipdb.set_trace()
# connect mappings; list for faster access
index_lookup = [output_map[input_map[p]] for p in sorted(list(input_map.keys()))]

# remap and write
output_file = open(roi_file_dataset_destination, 'w')
input_file = open(roi_file_a1, 'r')

for line in input_file.readlines():
    pieces = line.split(maxsplit=1)
    pieces[0] = str(index_lookup[int(pieces[0])])
    output_file.write(pieces[0] + " " + pieces[1])

input_file.close()
output_file.close()
print("Done!")
