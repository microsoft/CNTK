# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import zipfile
import cv2 # pip install opencv-python
import numpy as np
import os
import pdb

DEBUG = False
if DEBUG:
    import matplotlib.pyplot as mp


class ObjectDetectionReader:
    def __init__(self, img_map_file, roi_map_file, max_annotations_per_image,
                 pad_width, pad_height, pad_value, randomize, use_flipping,
                 max_images=None, buffered_rpn_proposals=None):
        self._pad_width = pad_width
        self._pad_height = pad_height
        self._pad_value = pad_value
        self._randomize = randomize
        self._use_flipping = use_flipping
        self._flip_image = True # will be set to False in the first call to _reset_reading_order
        self._buffered_rpn_proposals = buffered_rpn_proposals
        self._img_file_paths = []
        self._gt_annotations = []

        self._num_images = self._parse_map_files(img_map_file, roi_map_file, max_annotations_per_image, max_images)
        self._img_stats = [None for _ in range(self._num_images)]

        self._reading_order = None
        self._reading_index = -1
        
    def get_next_input(self):
        '''
        Reads image data and return image, annotations and shape information
        :return:
        img_data - The image data in CNTK format. The image is scale to fit into the size given in the constructor, centered and padded.
        roi_data - The ground truth annotations as numpy array of shape (max_annotations_per_image, 5), i.e. 4 coords + label per roi.
        img_dims - (pad_width, pad_height, scaled_image_width, scaled_image_height, orig_img_width, orig_img_height)
        '''

        index = self._get_next_image_index()
        roi_data = self._get_gt_annotations(index)
        if DEBUG:
            img_data, img_dims, resized_with_pad = self._load_resize_and_pad_image(index)
            self._debug_plot(resized_with_pad, roi_data)
        else:
            img_data, img_dims = self._load_resize_and_pad_image(index)
        buffered_proposals = self._get_buffered_proposals(index)

        return img_data, roi_data, img_dims, buffered_proposals

    def sweep_end(self):
        return self._reading_index >= self._num_images

    def _debug_plot(self, img_data, roi_data):
        color = (0, 255, 0)
        thickness = 2
        for rect in roi_data:
            pt1 = tuple([int(float(x)) for x in rect[0:2]])
            pt2 = tuple([int(float(x)) for x in rect[2:4]])
            try:
                cv2.rectangle(img_data, pt1, pt2, color, thickness)
            except:
                pdb.set_trace()
                print("Unexpected error:", sys.exc_info()[0])

        mp.imshow(img_data)
        mp.plot()
        mp.show()

    def _parse_map_files(self, img_map_file, roi_map_file, max_annotations_per_image, max_images):
        # read image map file and buffer sequence numbers
        with open(img_map_file) as f:
            img_map_lines = f.readlines()
        img_map_lines = [line for line in img_map_lines if len(line) > 0]
        if max_images is not None:
            img_map_lines = img_map_lines[:max_images]
        img_sequence_numbers = [int(x.split('\t')[0]) for x in img_map_lines]
        img_base_path = os.path.dirname(os.path.abspath(img_map_file))
        self._img_file_paths = [os.path.join(img_base_path, x.split('\t')[1]) for x in img_map_lines]

        # read roi map file
        with open(roi_map_file) as f:
            roi_map_lines = f.readlines()
        # TODO: check whether this avoids reading empty lines
        roi_map_lines = [line for line in roi_map_lines if len(line) > 0]
        if max_images is not None:
            roi_map_lines = roi_map_lines[:max_images]
        roi_sequence_numbers = []
        for roi_line in roi_map_lines:
            roi_sequence_numbers.append(int(roi_line[:roi_line.find(' ')]))
            rest = roi_line[roi_line.find(' ')+1:]
            bbox_input = rest[rest.find(' ')+1:]
            bbox_floats = np.fromstring(bbox_input, dtype=np.float32, sep=' ')
            num_floats = len(bbox_floats)
            assert num_floats % 5 == 0, "Ground truth annotation file is corrupt. Lines must contain 4 coordinates and a label per roi."
            annotations = np.zeros((max_annotations_per_image, 5))
            num_annotations = int(num_floats / 5)

            if num_annotations > max_annotations_per_image:
                print('Warning: The number of ground truth annotations ({}) is larger than the provided maximum number ({}).'
                      .format(num_annotations, max_annotations_per_image))
                bbox_floats = bbox_floats[:(max_annotations_per_image * 5)]
                num_annotations = max_annotations_per_image

            annotations[:num_annotations,:] = np.array(bbox_floats).reshape((num_annotations, 5))
            self._gt_annotations.append(annotations)

        # make sure sequence numbers match
        assert len(img_sequence_numbers) == len(roi_sequence_numbers), "number of images and annotation lines do not match"
        assert np.allclose(img_sequence_numbers, roi_sequence_numbers, 0, 0), "the sequence numbers in image and roi map files do not match"

        return len(img_sequence_numbers)

    def _reset_reading_order(self):
        self._reading_order = np.arange(self._num_images)
        if self._randomize:
            np.random.shuffle(self._reading_order)
        # if flipping should be used then we alternate between epochs from flipped to non-flipped
        self._flip_image = not self._flip_image if self._use_flipping else False

        self._reading_index = 0

    def _read_image(self, image_path):
        if "@" in image_path:
            at = str.find(image_path, '@')
            zip_file = image_path[:at]
            img_name = image_path[(at + 2):]
            archive = zipfile.ZipFile(zip_file, 'r')
            imgdata = archive.read(img_name)
            imgnp = np.array(bytearray(imgdata), dtype=np.uint8)
            img = cv2.imdecode(imgnp, 1)
        else:
            img = cv2.imread(image_path)

        return img

    def _prepare_annotations_and_image_stats(self, index, img_width, img_height):
        annotations = self._gt_annotations[index]
        do_scale_w = img_width > img_height
        target_w = self._pad_width
        target_h = self._pad_height

        if do_scale_w:
            scale_factor = float(self._pad_width) / float(img_width)
            target_h = int(np.round(img_height * scale_factor))
        else:
            scale_factor = float(self._pad_height) / float(img_height)
            target_w = int(np.round(img_width * scale_factor))

        top = int(max(0, np.round((self._pad_height - target_h) / 2)))
        left = int(max(0, np.round((self._pad_width - target_w) / 2)))
        bottom = self._pad_height - top - target_h
        right = self._pad_width - left - target_w

        xyxy = annotations[:, :4]
        xyxy *= scale_factor
        xyxy += (left, top, left, top)

        # not needed since xyxy is just a reference: annotations[:, :4] = xyxy
        # TODO: do we need to round/floor/ceil xyxy coords?
        annotations[:, 0] = np.round(annotations[:, 0])
        annotations[:, 1] = np.round(annotations[:, 1])
        annotations[:, 2] = np.round(annotations[:, 2])
        annotations[:, 3] = np.round(annotations[:, 3])

        # keep image stats for scaling and padding images later
        img_stats = [target_w, target_h, img_width, img_height, top, bottom, left, right]
        self._img_stats[index] = img_stats

    def _get_next_image_index(self):
        if self._reading_index < 0 or self._reading_index >= self._num_images:
            self._reset_reading_order()
        next_image_index = self._reading_order[self._reading_index]
        self._reading_index += 1
        return next_image_index

    def _load_resize_and_pad_image(self, index):
        image_path = self._img_file_paths[index]

        img = self._read_image(image_path)
        if self._img_stats[index] is None:
            img_width = len(img[0])
            img_height = len(img)
            self._prepare_annotations_and_image_stats(index, img_width, img_height)

        target_w, target_h, img_width, img_height, top, bottom, left, right = self._img_stats[index]

        resized = cv2.resize(img, (target_w, target_h), 0, 0, interpolation=cv2.INTER_NEAREST)
        resized_with_pad = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                              value=self._pad_value)
        if self._flip_image:
            resized_with_pad = cv2.flip(resized_with_pad, 1)

        # transpose(2,0,1) converts the image to the HWC format which CNTK accepts
        model_arg_rep = np.ascontiguousarray(np.array(resized_with_pad, dtype=np.float32).transpose(2, 0, 1))

        # dims = pad_width, pad_height, scaled_image_width, scaled_image_height, orig_img_width, orig_img_height
        dims = (self._pad_width, self._pad_height, target_w, target_h, img_width, img_height)
        if DEBUG:
            return model_arg_rep, dims, resized_with_pad
        return model_arg_rep, dims

    def _get_gt_annotations(self, index):
        annotations = self._gt_annotations[index]
        if self._flip_image:
            flipped_annotations = np.array(annotations)
            flipped_annotations[:,0] = self._pad_width - annotations[:,2] - 1
            flipped_annotations[:,2] = self._pad_width - annotations[:,0] - 1
            return flipped_annotations
        return annotations

    def _get_buffered_proposals(self, index):
        if self._buffered_rpn_proposals is None:
            return None

        buffered_proposals = self._buffered_rpn_proposals[index]
        if self._flip_image:
            flipped_proposals = np.array(buffered_proposals, dtype=np.float32)
            flipped_proposals[:,0] = self._pad_width - buffered_proposals[:,2] - 1
            flipped_proposals[:,2] = self._pad_width - buffered_proposals[:,0] - 1
            return flipped_proposals
        return buffered_proposals

