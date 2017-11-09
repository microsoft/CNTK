# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk.io import UserMinibatchSource, StreamInformation, MinibatchData
from cntk.core import Value
from utils.od_reader import ObjectDetectionReader
import numpy as np

class ObjectDetectionMinibatchSource(UserMinibatchSource):
    def __init__(self, img_map_file, roi_map_file, num_classes,
                 max_annotations_per_image, pad_width, pad_height, pad_value,
                 randomize, use_flipping, proposal_provider, proposal_iou_threshold=0.5,
                 provide_targets=False, normalize_means=None, normalize_stds=None, max_images=None):

        self.image_si = StreamInformation("image", 0, 'dense', np.float32, (3, pad_height, pad_width,))
        self.roi_si = StreamInformation("annotation", 1, 'dense', np.float32, (max_annotations_per_image, 5,))
        self.dims_si = StreamInformation("dims", 1, 'dense', np.float32, (4,))

        if proposal_provider is not None:
            num_proposals = proposal_provider.num_proposals()
            self.proposals_si = StreamInformation("proposals", 1, 'dense', np.float32, (num_proposals, 4))
            self.label_targets_si = StreamInformation("label_targets", 1, 'dense', np.float32, (num_proposals, num_classes))
            self.bbox_targets_si = StreamInformation("bbox_targets", 1, 'dense', np.float32, (num_proposals, num_classes*4))
            self.bbiw_si = StreamInformation("bbiw", 1, 'dense', np.float32, (num_proposals, num_classes*4))
        else:
            self.proposals_si = None

        self.od_reader = ObjectDetectionReader(img_map_file, roi_map_file, num_classes,
                                               max_annotations_per_image, pad_width, pad_height, pad_value,
                                               randomize, use_flipping, proposal_provider, proposal_iou_threshold,
                                               provide_targets, normalize_means, normalize_stds, max_images)

        super(ObjectDetectionMinibatchSource, self).__init__()

    def stream_infos(self):
        if self.proposals_si is None:
            return [self.image_si, self.roi_si, self.dims_si]
        else:
            return [self.image_si, self.roi_si, self.dims_si, self.proposals_si, self.label_targets_si, self.bbox_targets_si, self.bbiw_si]

    def image_si(self):
        return self.image_si

    def roi_si(self):
        return self.roi_si

    def dims_si(self):
        return self.dims_si

    def proposals_si(self):
        return self.proposals_si

    def label_targets_si(self):
        return self.label_targets_si

    def bbox_targets_si(self):
        return self.bbox_targets_si

    def bbiw_si(self):
        return self.bbiw_si

    def next_minibatch(self, num_samples, number_of_workers=1, worker_rank=1, device=None, input_map=None):
        if num_samples > 1:
            print("Only single item mini batches are supported currently by od_mb_source.py")
            exit(1)

        img_data, roi_data, img_dims, proposals, label_targets, bbox_targets, bbox_inside_weights = self.od_reader.get_next_input()
        sweep_end = self.od_reader.sweep_end()

        if input_map is None:
            result = {
                self.image_si: MinibatchData(Value(batch=img_data), 1, 1, sweep_end),
                self.roi_si: MinibatchData(Value(batch=roi_data), 1, 1, sweep_end),
                self.dims_si: MinibatchData(Value(batch=np.asarray(img_dims, dtype=np.float32)), 1, 1, sweep_end),
                self.proposals_si: MinibatchData(Value(batch=np.asarray(proposals, dtype=np.float32)), 1, 1, sweep_end),
                self.label_targets_si: MinibatchData(Value(batch=np.asarray(label_targets, dtype=np.float32)), 1, 1, sweep_end),
                self.bbox_targets_si: MinibatchData(Value(batch=np.asarray(bbox_targets, dtype=np.float32)), 1, 1, sweep_end),
                self.bbiw_si: MinibatchData(Value(batch=np.asarray(bbox_inside_weights, dtype=np.float32)), 1, 1, sweep_end),
            }
        else:
            result = {
                input_map[self.image_si]: MinibatchData(Value(batch=np.asarray(img_data, dtype=np.float32)), 1, 1, sweep_end)
            }
            if self.roi_si in input_map:
                result[input_map[self.roi_si]] = MinibatchData(Value(batch=np.asarray(roi_data, dtype=np.float32)), 1, 1, sweep_end)
            if self.dims_si in input_map:
                result[input_map[self.dims_si]] = MinibatchData(Value(batch=np.asarray(img_dims, dtype=np.float32)), 1, 1, sweep_end)
            if self.proposals_si in input_map:
                result[input_map[self.proposals_si]] = MinibatchData(Value(batch=np.asarray(proposals, dtype=np.float32)), 1, 1, sweep_end)
            if self.label_targets_si in input_map:
                result[input_map[self.label_targets_si]] = MinibatchData(Value(batch=np.asarray(label_targets, dtype=np.float32)), 1, 1, sweep_end)
            if self.bbox_targets_si in input_map:
                result[input_map[self.bbox_targets_si]] = MinibatchData(Value(batch=np.asarray(bbox_targets, dtype=np.float32)), 1, 1, sweep_end)
            if self.bbiw_si in input_map:
                result[input_map[self.bbiw_si]] = MinibatchData(Value(batch=np.asarray(bbox_inside_weights, dtype=np.float32)), 1, 1, sweep_end)

        return result
