# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk import output_variable
from cntk.ops.functions import UserFunction
import numpy as np
import cv2
import matplotlib.pyplot as mp
import copy, textwrap
from PIL import Image, ImageFont, ImageDraw
from PIL.ExifTags import TAGS


import pdb

class DebugLayer(UserFunction):
    def __init__(self, arg1, arg2, arg3, name='DebugLayer', debug_name=""):
        super(DebugLayer, self).__init__([arg1, arg2, arg3], name=name)
        self._debug_name = debug_name

    def infer_outputs(self):
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes)]

    def forward(self, arguments, device=None, outputs_to_retain=None):
        outputs = arguments[0].copy()
        images = arguments[1].copy()
        gts = arguments[2].copy()

        num_images = images.shape[0]
        for i in np.arange(num_images):
            img_data = images[i]
            img_data_transposed = img_data.transpose(1, 2, 0)

            gt_boxes = gts[i]
            gt_boxes.shape = (3, 5)

            ih = img_data_transposed.shape[0]
            iw = img_data_transposed.shape[1]
            self._boxes_to_absolute_xyxy(gt_boxes, iw, ih)

            # responsible boxes [527, 527], [453, 453, 523, 523, 630, 630]
            net_boxes = outputs[i,:,:5]
            self._boxes_to_absolute_xyxy(net_boxes, iw, ih)

            #self._visualize_image(img_data_transposed, gt_boxes, "ground truth")
            self._visualize_image(img_data_transposed, net_boxes, "net output")


        print("-- {} -- shapes".format(self._debug_name))
        print("image shape {}".format(images.shape))
        print("output shape {}".format(outputs.shape))
        print("gt shape {}".format(gts.shape))

        return None, arguments[0]

    def backward(self, state, root_gradients, variables):
        if self.inputs[0] in variables:
            variables[self.inputs[0]] = root_gradients

    def clone(self, cloned_inputs):
        return DebugLayer(cloned_inputs[0], cloned_inputs[1], cloned_inputs[2], debug_name=self._debug_name)

    def serialize(self):
        internal_state = {}
        internal_state["debug_name"] = self._debug_name
        return internal_state

    @staticmethod
    def deserialize(inputs, name, state):
        debug_name = state['debug_name']
        return DebugLayer(inputs[0], inputs[1], inputs[2], name=name, debug_name=debug_name)

    def _boxes_to_absolute_xyxy(self, boxes, iw, ih):
        boxes[:, 0] *= iw
        boxes[:, 1] *= ih
        hbw = boxes[:, 2] * (iw / 2)
        hbh = boxes[:, 3] * (ih / 2)
        boxes[:, 2] = boxes[:, 0] + hbw
        boxes[:, 3] = boxes[:, 1] + hbh
        boxes[:, 0] = boxes[:, 0] - hbw
        boxes[:, 1] = boxes[:, 1] - hbh

    def _visualize_image(self, image_data, box_coords, title, decisionThreshold = 0.5):
        imgDebug = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        img_widht = image_data.shape[1]
        img_height = image_data.shape[0]

        scores = box_coords[:,4].copy()
        sorted_scores = np.sort(scores)
        a = 5
        b = 30
        top_a_score = sorted_scores[-a] if len(scores) > a else 0.0
        top_b_score = sorted_scores[-b] if len(scores) > b else 0.0

        # draw multiple times to avoid occlusions
        for roiIndex in range(len(box_coords)):
            rect = box_coords[roiIndex,:4]
            score = box_coords[roiIndex,4]
            if score < top_b_score: continue
            label = score > top_a_score

            # init drawing parameters
            thickness = 4 if label else 1
            color = (255, 100, 100)

            rect[0] = int(max(0, min(img_widht, rect[0])))
            rect[1] = int(max(0, min(img_height, rect[1])))
            rect[2] = int(max(0, min(img_widht, rect[2])))
            rect[3] = int(max(0, min(img_height, rect[3])))

            self._drawRectangles(imgDebug, [rect], color=color, thickness=thickness)
            #text = "(" + str(round(score, 3 )) + ")"
            #imgDebug = self._drawText(imgDebug, (rect[0], rect[1]), text, color=(255, 255, 255), colorBackground=color)

        mp.imshow(imgDebug)
        mp.plot()
        mp.show()

    def _drawRectangles(self, img, rects, color = (0, 255, 0), thickness = 2):
        for rect in rects:
            pt1 = tuple([int(x) for x in rect[0:2]])
            pt2 = tuple([int(x) for x in rect[2:]])
            try:
                cv2.rectangle(img, pt1, pt2, color, thickness)
            except:
                import pdb; pdb.set_trace()
                print("Unexpected error:", sys.exc_info()[0])

    def _drawText(self, img, pt, text, textWidth=None, color = (255,255,255), colorBackground = None, font = ImageFont.truetype("arial.ttf", 16)):
        pilImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        rgb = pilImg.convert('RGB')
        return np.array(rgb).copy()[:, :, ::-1]
