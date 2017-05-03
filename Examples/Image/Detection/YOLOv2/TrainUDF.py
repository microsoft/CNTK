# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================


from cntk.ops.functions import UserFunction
from cntk.ops import *
import numpy as np
import math

from PARAMETERS import *


class TrainFunction(UserFunction):


    def __init__(self,
                 arg1, gtbs,
                 grid_size_hor=(par_image_width/par_downsample),
                 grid_size_ver=(par_image_height / par_downsample),
                 num_anchorboxes=par_num_anchorboxes,
                 anchorbox_scales=par_anchorbox_scales,
                 lambda_coord =5.0,
                 lambda_no_obj = 0.5,
                 name="TrainFunction"):
        super(TrainFunction, self).__init__([arg1, gtbs], name=name)
        self.grid_size_hor = grid_size_hor
        self.grid_size_ver = grid_size_ver
        self.num_anchorboxes = num_anchorboxes
        self.lambda_no_obj = lambda_no_obj
        self.lambda_coord = lambda_coord
        self.anchorbox_scales  = anchorbox_scales

    def forward(self, arguments, outputs=None, keep_for_backward=None, device=None, as_numpy=False):
        goal, scale = self.create_outputs(arguments[0], arguments[1])

        outputs[self.outputs[0]] = goal
        outputs[self.outputs[1]] = scale
        outputs[self.outputs[2]] = self.make_wh_sqrt(network=arguments[0])
        return None

    def backward(self, state, root_gradients, variables):
        assert False , "backward should not be called!"
        return None


    def infer_outputs(self):
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes, needs_gradient=False),
                output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes, needs_gradient=False),
                output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes, needs_gradient=True)]

    def serialize(self):
        internal_state = {}
        internal_state['grid_size_hor'] = self.grid_size_hor
        internal_state['grid_size_ver'] = self.grid_size_ver
        internal_state['num_anchorboxes'] = self.num_anchorboxes
        internal_state['anchorbox_scales'] = self.anchorbox_scales
        internal_state['lambda_no_obj'] = self.lambda_no_obj
        internal_state['lambda_coord'] = self.lambda_coord
        return internal_state

    @staticmethod
    def deserialize(inputs, name, state):
        # im_info = state['im_info']
        grid_size_hor = state['grid_size_hor']
        grid_size_ver = state['grid_size_ver']
        num_anchorboxes = state['num_anchorboxes']
        anchorbox_scales = state['anchorbox_scales']
        lambda_no_obj = state['lambda_no_obj']
        lambda_coord = state['lambda_coord']
        return TrainFunction(inputs[0], inputs[1],
                             grid_size_hor=grid_size_hor, grid_size_ver=grid_size_ver,
                             num_anchorboxes=num_anchorboxes, anchorbox_scales=anchorbox_scales,
                             lambda_coord=lambda_coord, lambda_no_obj=lambda_no_obj,
                             name=name)

    def create_outputs(self, eval_results, gtb_inputs):
        # get dimensions
        mb_size = eval_results.shape[0]
        num_vectors = eval_results.shape[1]
        assert num_vectors == self.grid_size_hor * self.grid_size_ver * self.num_anchorboxes, "number of vectors do not match"
        vector_length = eval_results.shape[2]
        assert vector_length > 5, "Vector is too short! Length must be >5 (x,y,w,h,obj,cls...)"
        num_cls = vector_length - 5
        # remove asserions when tested?

        list_scales = [[[0]*4+[self.lambda_no_obj] + [0]*num_cls]*num_vectors]*mb_size
        # set default values first so not every position needs to be written manually
        array_goal = np.zeros(eval_results.shape)
        array_scale = np.asarray(list_scales)


        gtb_array = gtb_inputs.reshape((len(gtb_inputs)/5 , 5)) #shape (50,5)
        # Here we are going completely for numpy and for loops and no cntk/parallised ops! TODO Switch to CNTK implementation
        for i in (len(gtb_array)):
            print (i) # TODO remove
            #find gridcells who are inside of the gtb
            left_gc_index = int(gtb_array[i][1] - 0.5 * gtb_array[i][3] * self.grid_size_vhor)
            left_gc_index = math.max(left_gc_index, 0)
            right_gc_index =int(gtb_array[i][1] + 0.5 * gtb_array[i][3] * self.grid_size_hor)
            right_gc_index = math.min(right_gc_index, self.grid_size_hor)

            top_gc_index =  int(gtb_array[i][1] - 0.5 * gtb_array[i][3] * self.grid_size_ver)
            top_gc_index = math.max(top_gc_index, 0)
            bottom_gc_index = int(gtb_array[i][1] + 0.5 * gtb_array[i][3] * self.grid_size_ver)
            bottom_gc_index = math.min(bottom_gc_index, self.grid_size_ver)

            for x in range(left_gc_index, right_gc_index + 1):
                for y in range (top_gc_index, bottom_gc_index +1 ):
                    highest_iou_index = None
                    highest_iou = 0
                    for z in range (self.num_anchorboxes):
                        # Set Pr(Class|Obj) for the class found to 1
                        # if a gc has multipla classes ALL are set!
                        array_goal[y * self.grid_size_ver * self.num_anchorboxes + x * self.num_anchorboxes + z][5 + gtb_array[i][0]] = 1
                        for cls_sc_i in range(num_cls):
                            array_scale[y * self.grid_size_ver * self.num_anchorboxes + x * self.num_anchorboxes + z][5+cls_sc_i] = 1

                        #score iou to find bb
                        ab_iou = self.calc_iou((gtb_array[i][0],gtb_array[i][1],gtb_array[i][2],gtb_array[i][3]),
                                               (x * 1.0/self.grid_size_hor, x * 1.0/self.grid_size_ver, self.anchorbox_scales[z][0], self.anchorbox_scales[z][1]))
                        # save iou to determine whether ab contained the obj or noobj constant needs to be applied!
                        array_goal[y * self.grid_size_ver * self.num_anchorboxes + x * self.num_anchorboxes + z][4] = ab_iou

                        if ab_iou > highest_iou:
                            highest_iou = ab_iou
                            highest_iou_index = z

                    vector_index = y * self.grid_size_ver * self.num_anchorboxes + x * self.num_anchorboxes + highest_iou_index
                    if(highest_iou_index is not None and highest_iou > array_goal[vector_index][4]):
                        # if an ab is in scope and no other gtb with higher iou has jet trained this ab
                        array_goal[vector_index][0] = gtb_array[i][1]
                        array_goal[vector_index][1] = gtb_array[i][2]
                        array_goal[vector_index][2] = gtb_array[i][3]
                        array_goal[vector_index][3] = gtb_array[i][4]
                        array_goal[vector_index][4] = highest_iou

                        for z in range(4):
                            array_scale[vector_index][z] = self.lambda_coord

                        for z in range(self.num_anchorboxes):
                            curr_index = y * self.grid_size_ver * self.num_anchorboxes + x * self.num_anchorboxes + z
                            if(array_goal[curr_index][4] > 0):
                                # TODO scale is set to 0 for other bounding boxes containing the gtb. In the paper it is unclear whether scale should be 0 or lambda_no_obj and goal= 0 (or = iou?)
                                array_scale[curr_index] = 1# if curr_index == vector_index else 0

        #   defaulft= set scale(x,y,w,h,cls)=0, set scale(obj)=lambda_no_obj, set goal(obj)=0 (set goal(*)=0)
        #   find gridcells who are inside of the gtb
        #      batch?: set the cls scale to 1, set goal to 0default and 1 where class
        #              set scale(obj) = 1, goal(obj)=IOU (might have to be done single_op)
        #      single_op: find anchorboxes which should be trained
        #         set x,y,w,h of those (goal=gtb, scale = lambda_cord)
        #         set objness of those goal=1, scale =1
        #


        return array_goal, array_scale

    @staticmethod
    def calc_iou(coords1, coords2): #coords are tuples (x,y,w,h)
        area1 = coords1[2] * coords1[3] # width * height
        area2 = coords2[2] * coords2[3]
        assert  area1 >= 0,"Area 1 must not be below 0"
        assert area2 >= 0, "Area 2 must not be below 0"

        # intersection
        intersection = 0
        xl1 = coords1[0] - coords1[2]/2
        xr1 = coords1[0] + coords1[2] / 2
        xl2 = coords2[0] - coords2[2] / 2
        xr2 = coords2[0] + coords2[2] / 2
        yt1 = coords1[1] - coords1[3] / 2
        yb1 = coords1[1] + coords1[3] / 2
        yt2 = coords2[1] - coords2[3] / 2
        yb2 = coords2[1] + coords2[3] / 2

        #get left bound of iou
        if xl1 <= xl2 <= xr1 :
            left = xl2
        elif xl2 <= xl1 <= xr2:
            left = xl1

        if xl1 <= xr2 <= xr1:
            right = xr2
        elif xl2 <= xr1 <= xr2:
            right = xr1

        if yt1 <= yt2 <= yb1 :
            top = yt2
        elif yt2 <= yt1 <= yb2:
            top = yt1

        if yt1 <= yb2 <= yb1:
            bottom = yb2
        elif yt2 <= yb1 <= yb2:
            bottom = yb1

        if left is not None and right is not None and top is not None and bottom is not None:
            intersection = (right - left) * (bottom - top)
        else:
            intersection = 0
        assert intersection >= 0, "Intersection must be equal or greater 0"

        return intersection / (area1 + area2 - intersection)

    @staticmethod
    def make_wh_sqrt(network):
        xy = network[:,:,0:2]
        wh = network[:,:,2:4]
        rest=network[:,:,4:]

        sqrt_wh = sqrt(wh)
        spliced = splice(xy,sqrt_wh,rest,axis=3)

        return spliced
