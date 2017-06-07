# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================


from cntk import user_function
from cntk.ops import *
import numpy as np
from PARAMETERS import *
from TrainUDFyolov2 import TrainFunction

def get_error(network, gtb_input, cntk_only=False):
    if cntk_only:
        err_f = ErrorFunction()
        return err_f.evaluate_network(network, gtb_input)

    else:
        ud_tf = TrainFunction(network, gtb_input)
        training_model = user_function(ud_tf)

        #err = TrainFunction.make_wh_sqrt(training_model.outputs[0]) - TrainFunction.make_wh_sqrt(network)
        err = alias(training_model.outputs[0], 'TrainFunction_0') - network
        sq_err = err * err
        sc_err = sq_err * alias(training_model.outputs[1], 'TrainFunction_1')  # apply scales (lambda_coord, lambda_no_obj, zeros on not learned params)
        mse = reduce_sum(sc_err, axis=Axis.all_static_axes(), name="MeanSquaredError")
        return mse

def test_get_error():
    """
    Test for get_error()
    :return: Nothing
    """
    assert False, "Not implemented yet"

"""
# The class below is an approach to use only cntk ops instead of the user defined function. It does not work well though and is not the preferred model because of that.
class ErrorFunction(): # CNTK-only error function
    def __init__(self,
                 grid_size_hor=(par_image_width/par_downsample),
                 grid_size_ver=(par_image_height / par_downsample),
                 num_anchorboxes=par_num_anchorboxes,
                 anchorbox_scales=par_anchorbox_scales,
                 num_gtbs_per_input=par_max_gtbs,
                 lambda_coord =5.0,
                 lambda_no_obj = 0.5,
                 num_classes = par_num_classes,
                 name="ErrorFunction"):

        self.grid_size_hor = int(grid_size_hor)
        self.grid_size_ver = int(grid_size_ver)
        self.num_anchorboxes = int(num_anchorboxes)
        assert lambda_no_obj <= 1, "lambda_no_obj must be smaller or equal 1"
        self.lambda_no_obj = lambda_no_obj
        self.lambda_coord = lambda_coord
        self.anchorbox_scales  = anchorbox_scales
        self.num_gtbs_per_input = int(num_gtbs_per_input)
        self.num_classes = num_classes
        self.create_constants()

    def create_constants(self):
        self.ab_scales_const = constant(np.ascontiguousarray(self.anchorbox_scales, dtype=np.float32))
        self.scale_xywho_responsible = constant(np.ascontiguousarray([self.lambda_coord]*4 + [1]), dtype=np.float32)
        self.scale_xywho_no_obj = constant(np.ascontiguousarray([0]*4 + [self.lambda_no_obj]), dtype=np.float32)
        self.scale_cls_array = constant(np.ascontiguousarray([[1]*self.num_classes]*self.num_anchorboxes))

        grid_const_map = {}
        grid_const_map["grid_xmin_0"] = constant(np.ascontiguousarray([0]), dtype=np.float32)
        for x in range(1, self.grid_size_hor):
            new_const = constant(np.ascontiguousarray([x*1.0/self.grid_size_hor]), dtype=np.float32)
            grid_const_map["grid_xmin_" + str(x)] = new_const
            grid_const_map["grid_xmax_" + str(x-1)] = new_const
        grid_const_map["grid_xmax_" + str(self.grid_size_hor-1)] = constant(np.ascontiguousarray([1]), dtype=np.float32)

        grid_const_map["grid_ymin_0"] = constant(np.ascontiguousarray([0]), dtype=np.float32)
        for y in range(1, self.grid_size_hor):
            new_const = constant(np.ascontiguousarray([y * 1.0 / self.grid_size_hor]), dtype=np.float32)
            grid_const_map["grid_ymin_" + str(y)] = new_const
            grid_const_map["grid_ymax_" + str(y - 1)] = new_const
        grid_const_map["grid_ymax_" + str(self.grid_size_hor - 1)] = constant(np.ascontiguousarray([1]),
                                                                              dtype=np.float32)

        self.grid_const_map = grid_const_map


    def handle_gridcell_for_gtb(self, x, y, gtb, local_predicted_bbs):
        xy_gtb = gtb[:2]
        wh_gtb = gtb[2:4]
        gtb_cls_vector = one_hot(gtb[4:5] - 1,self.num_classes)

        ######## RESPONISIBILITIES #########
        # check wether gc is responsible for gtb
        xmin = self.grid_const_map["grid_xmin_" + str(x)]
        xmax = self.grid_const_map["grid_xmax_" + str(x)]
        gtb_x = gtb[0:1]
        gc_is_respondible_x = greater_equal(gtb_x, xmin) * less_equal(gtb_x, xmax)

        ymin = self.grid_const_map["grid_ymin_" + str(y)]
        ymax = self.grid_const_map["grid_ymax_" + str(y)]
        gtb_y = gtb[1:2]
        gc_is_respondible_y = greater_equal(gtb_x, xmin) * less_equal(gtb_x, xmax)

        gc_is_respondible = gc_is_respondible_x * gc_is_respondible_y  # shape = (1,)

        gtb_w_half = gtb[2:3] / 2
        gtb_h_half = gtb[3:4] / 2
        # check whether gtb contains (at least a part of) the gc (shape = 1)
        # left bound of gtb smaller than right bound gc  AND
        # right bound of gtb greater than left bound gc  AND
        # bottom bound of gtb smaller than top bound gc  AND
        # top bound of gtb greater than bottom bound gc  AND
        gtb_contains_gc = less(gtb_x - gtb_w_half, xmax) \
                          * greater(gtb_x + gtb_w_half, xmin) \
                          * less(gtb_y - gtb_h_half, ymax) \
                          * greater(gtb_y + gtb_h_half, ymin)

        xy_consts = splice(*[reshape(splice(xmin + (.5 / self.grid_size_hor), ymin + (.5 / self.grid_size_ver)),
                                     (1, 2))] * self.num_anchorboxes, axis=0)
        # create abox tuples for iou calculation
        ab_boxes = splice(xy_consts, self.ab_scales_const, axis=1)
        gtb_xywh = reshape(gtb[0:4], (1, 4))
        gtbox_array = splice(*[gtb_xywh] * self.num_anchorboxes, axis=0)

        # calc ious
        ab_ious = self.multi_iou(ab_boxes, gtbox_array)
        # edge cases --> all ious 0, gtb not applying for cell etc! --> via gc_is_responsible ? #TODO check for other edge cases
        # classes among all aboxes! done
        # done: calculate actual iou with real box for train val! TARGET(obj) IS NOT THE AB-IOU BUT THE ACTUAL ONE
        # find and fill only resposible box
        index = argmax(ab_ious)
        responsibility_vector = reshape(one_hot(index, self.num_anchorboxes), (self.num_anchorboxes, 1))

        ####### Targets #########
        actual_ious = self.multi_iou(gtbox_array, local_predicted_bbs)
        iou_with_pred_resp = reshape(reshape(actual_ious, (1, 5)) @ responsibility_vector, (1,))

        xywho_vector = reshape(splice(xy_gtb, wh_gtb, iou_with_pred_resp), (1, 5)) * gc_is_respondible
        xywho_target = responsibility_vector @ xywho_vector

        cls_vector = gtb_cls_vector * gtb_contains_gc
        cls_target = splice(*[reshape(cls_vector, (1,self.num_classes))] * self.num_anchorboxes, axis=0)

        local_target = splice(xywho_target, cls_target, axis=1)

        ######### Scales ########

        # if gc_is_responsible == 1 then gtb_contains_gc must be 1! --> no_obj case only when gbt_contains_gc = 0  (gc_is_responsible must be 0 then)
        local_scale_xywho_obj = (responsibility_vector @ reshape(self.scale_xywho_responsible * gc_is_respondible, (1, 5)))
        local_scale_xywho_no_obj = (
        splice(*[reshape(self.scale_xywho_no_obj, (1, 5))] * self.num_anchorboxes, axis=0) * (1 - gtb_contains_gc))
        local_scale_xywho = local_scale_xywho_obj + local_scale_xywho_no_obj
        local_scale_cls = self.scale_cls_array * gtb_contains_gc
        local_scales = splice(local_scale_xywho, local_scale_cls, axis=1)

        # local_scales = local_target * 0
        return local_target, local_scales


    def handle_single_gtb(self, gtb, predicted_bbs):
        targets = []
        scales = []

        for y in range(self.grid_size_ver):
            for x in range(self.grid_size_hor):
                curr_bbs = predicted_bbs[(y * self.grid_size_hor + x) * self.num_anchorboxes: (
                                                                                              y * self.grid_size_hor + x + 1) * self.num_anchorboxes]
                local_target, local_scale = self.handle_gridcell_for_gtb(x, y, gtb, curr_bbs)
                targets.append(local_target)
                scales.append(local_scale)

                # if targets is None:
                #    targets = local_target
                #    scales = local_scale
                # else:
                #    targets = splice(targets, local_target, axis=0)
                #    scales = splice(scales, local_scale, axis=0)

        gtb_target = splice(*targets, axis=0)
        gtb_scales = splice(*scales, axis=0)

        return gtb_target, gtb_scales


    def combine_gtb_slices(self, target_list, scale_list):
        const_output_width = self.num_anchorboxes * self.grid_size_hor * self.grid_size_ver
        const_output_height =self.num_classes + 5

        new_targetlist = [];
        new_scalelist = []
        for i, curr_target in enumerate(target_list):
            new_shape = (1, curr_target.shape[0], curr_target.shape[1])  # TODO
            new_targetlist.append(reshape(curr_target, new_shape))
            new_scalelist.append(reshape(scale_list[i], new_shape))

        # shape = (num_gtbs, num_gc_hor * num_gc_ver * num_ab, 5+num_cls)
        target_block = splice(*new_targetlist, axis=0)
        scale_block = splice(*new_scalelist, axis=0)

        # Reduce Targets
        # index = argmax(obj)
        # take only that slice for x,y,w,h
        indicies = argmax(target_block[:, :, 4:5], axis=0)
        one_hot_flat = one_hot(reshape(indicies, (1, const_output_width)), self.num_gtbs_per_input)
        selector_vol = splice(*[transpose(one_hot_flat, (2, 1, 0))] * 5,
                              axis=2)  # TODO ASK: flip splice and transpose for performance?
        sel_targets_upper = target_block[:, :, 0:5] * selector_vol
        target_upper = reshape(reduce_max(sel_targets_upper, axis=0), (const_output_width, 5))
        # for cls sum and div by max
        sum_cls = reshape(reduce_sum(target_block[:, :, 5:], axis=0), (const_output_width,self.num_classes))
        cls_divisor = splice(*[reduce_sum(sum_cls, axis=1)] *self.num_classes,
                             axis=1)  # scale the sum of each class vector to 1 for softmax!
        target_lower = sum_cls / element_min(cls_divisor, 1)  # prevent divion by 0!

        target = splice(target_upper, target_lower, axis=1)

        assert target.shape == (const_output_width, const_output_height)

        # reduce scale vector
        # - x,y,w,h {0, lambda_coord>0} --> select max to loose nothing
        # - obj {0?, 1, 1>lambda_no_obj>0} --> select 1 if any, else max! regarding lambda_no_obj <= 1 --> select max
        # - cls {0, 1} --> select max for if one gtb applies
        scale = reshape(reduce_max(scale_block, axis=0), (const_output_width, const_output_height))

        return target, scale


    def create_targets_and_scale(self, gtb_inputs, predicted_bbs):
        target_list = []
        scale_list = []

        for i in range(self.num_gtbs_per_input):
            print(i)
            # slice_target, slice_scale = self.handle_single_gtb(reshape(gtb_inputs[i:i+1],(5,)), predicted_bbs)
            slice_target, slice_scale = self.handle_single_gtb(reshape(gtb_inputs[i], (5,)), predicted_bbs)
            target_list.append(slice_target)
            scale_list.append(slice_scale)

        target, scale = self.combine_gtb_slices(target_list, scale_list)

        return stop_gradient(target), stop_gradient(scale)  # stop gradients on path w/o params


    def evaluate_network(self, network, gtb_inputs):
        gtbs_arranged = reshape(gtb_inputs, (self.num_gtbs_per_input, 5))
        network_bbs = network[:, 0:4]
        target, scale = self.create_targets_and_scale(gtbs_arranged, network_bbs)

        error = network - target
        scaled_squared_error = (error * error) * scale
        mse = reduce_mean(scaled_squared_error)

        return mse

    @staticmethod
    def multi_iou(coords1, coords2):
        '''
        Calulates the IOU of the boxes specified in coords1 and coords2.
        :param coords1: Must have the shape (number_of_boxes, box_coordinates), where number_of boxes of coords1 and coords2 must be equal. box_xoordinates are specified as [X, Y, W, H], where X and Y describe the X- and Y-Coordinate of the boxes center and W and H describe the boxes width and height.
        :param coords2: See coord1
        :return: Output of shape (number_of_boxes, 1) with the element at index i specified by the IOU of the boxes described by coords1[i] and coords2[i].
        '''

        # assert coords1.shape() == coords2.shape()

        w1 = coords1[:, 2:3]
        w2 = coords2[:, 2:3]
        h1 = coords1[:, 3:4]
        h2 = coords2[:, 3:4]

        x1 = coords1[:, 0:1]
        x2 = coords2[:, 0:1]
        y1 = coords1[:, 1:2]
        y2 = coords2[:, 1:2]

        xmin_inter = element_max((x1 - (w1 / 2)),
                                 (x2 - (w2 / 2)))  # get left bound of possible iou area (right one of the lef´t bounds)
        xmax_inter = element_min((x1 + (w1 / 2)),
                                 (x2 + (w2 / 2)))  # get left bound of possible iou area (left one of the right bounds)
        w_inter = relu(xmax_inter - xmin_inter)  # calc width; if width is negative there is no intersection --> set 0

        ymin_inter = element_max((y1 - (h1 / 2)), (y2 - (h2 / 2)))
        ymax_inter = element_min((y1 + (h1 / 2)), (y2 + (h2 / 2)))
        h_inter = relu(ymax_inter - ymin_inter)

        intersection = w_inter * h_inter

        # max_intersex_x = element_min(w1,w2)
        # max_intersex_y = element_min(h1,h2)
        # dist_x = abs(x1 - x2)
        # dist_y = abs(y1 - y2)
        # unconstrained_intersec_x = (w1 + w2) / 2  - dist_x
        # unconstrained_intersec_y = (h1 + h2) / 2 - dist_y
        # intersec_x = element_min(unconstrained_intersec_x, max_intersex_x) # set maximum bounds in case one box is completely inside of the other
        # intersec_y = element_min(unconstrained_intersec_y, max_intersex_y)
        # intersection = relu(intersec_x) * relu(intersec_y)

        area1 = w1 * h1
        area2 = w2 * h2
        union = (area1 + area2) - intersection

        return intersection / union
"""
