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

class LambdaFunc(UserFunction): # usefoll for debugging...
    def __init__(self,
            arg,
            when=lambda arg: True,
            execute=lambda arg: print(arg.shape),
            name=''):
        self.when = when
        self.execute = execute

        super(LambdaFunc, self).__init__([arg], name=name)

    def infer_outputs(self):
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes)]

    def forward(self, argument, device=None, outputs_to_retain=None):
        if self.when(argument):
            self.execute(argument)

        return None, argument

    def backward(self, state, root_gradients):
        return root_gradients

    def clone(self, cloned_inputs):
        return self.__class__(*cloned_inputs, name=self.name)

class TrainFunction(UserFunction):

    ####### Constructor #######
    def __init__(self,
                 arg1, gtbs,
                 grid_size_hor=(par_image_width/par_downsample),
                 grid_size_ver=(par_image_height / par_downsample),
                 num_anchorboxes=par_num_anchorboxes,
                 anchorbox_scales=par_anchorbox_scales,
                 num_gtbs_per_img=par_max_gtbs,
                 lamda_coord =par_lamda_coord,
                 lamda_obj=par_lamda_obj,
                 lamda_no_obj = par_lamda_no_obj,
                 lamda_cls = par_lamda_cls,
                 objectness_threshold=par_objectness_threshold,
                 default_box_values_for_first_n_mb = par_box_default_mbs,
                 default_box_scale=par_scale_default_boxes,
                 name="TrainFunction"):
        super(TrainFunction, self).__init__([arg1, gtbs], name=name)

        self.grid_size_hor = int(grid_size_hor)
        self.grid_size_ver = int(grid_size_ver)

        assert lamda_no_obj <= 1, "lambda_no_obj must be smaller or equal 1"
        self.lambda_no_obj = lamda_no_obj
        self.lambda_coord = lamda_coord
        self.lamda_obj = lamda_obj
        self.lamda_cls = lamda_cls
        self.objectness_threshold=objectness_threshold
        self.num_anchorboxes = int(num_anchorboxes)
        self.anchorbox_scales  = anchorbox_scales

        self.num_gtbs_per_img = int(num_gtbs_per_img)

        self.default_box_values_for_first_n_mb=default_box_values_for_first_n_mb
        self.default_box_scale=default_box_scale

        self.loggingArray=np.zeros((self.num_anchorboxes, self.grid_size_hor, self.grid_size_ver))
        self.logCount = 1
        self.mb_count=0

    def logHit(self,x,y,z):
        self.loggingArray[z,x,y]+=1
        if(self.logCount % 1000 == 0):
            print("Logged hits of last 1000 samples")
            print(self.loggingArray)
            self.loggingArray = np.zeros(self.loggingArray.shape)
            self.logCount=0
        self.logCount+=1


    ######### @Overrides #########
    # @Override
    def forward(self, arguments, outputs=None, keep_for_backward=None, device=None, as_numpy=False):
        targets, scales = self.create_outputs_like_cyolo(arguments[0], arguments[1])
        #import ipdb;ipdb.set_trace()
        outputs[self.outputs[0]] = np.ascontiguousarray(targets, np.float32)
        outputs[self.outputs[1]] = np.ascontiguousarray(scales, np.float32)

        if False:
            self.check_values(arguments[0], outputs[self.outputs[0]], outputs[self.outputs[1]])

        return None

    # @Override
    def backward(self, state, root_gradients, variables):
        return None

    # @Override
    def infer_outputs(self):
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes, needs_gradient=False),
                output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes, needs_gradient=False)]

    # @Override
    def serialize(self):
        internal_state = {}
        internal_state['grid_size_hor'] = self.grid_size_hor
        internal_state['grid_size_ver'] = self.grid_size_ver
        internal_state['num_anchorboxes'] = self.num_anchorboxes
        internal_state['anchorbox_scales'] = self.anchorbox_scales
        internal_state['lambda_no_obj'] = self.lambda_no_obj
        internal_state['lambda_coord'] = self.lambda_coord
        internal_state['num_gtbs_per_img'] = self.num_gtbs_per_img

        internal_state['lamda_obj'] = self.lamda_obj
        internal_state['lamda_cls'] = self.lamda_cls
        internal_state['objectness_threshold'] = self.objectness_threshold
        internal_state['default_box_values_for_first_n_mb'] = self.default_box_values_for_first_n_mb
        internal_state['default_box_scale'] = self.default_box_scale
        return internal_state

    # @Override
    @staticmethod
    def deserialize(inputs, name, state):
        # im_info = state['im_info']
        grid_size_hor = state['grid_size_hor']
        grid_size_ver = state['grid_size_ver']
        num_anchorboxes = state['num_anchorboxes']
        anchorbox_scales = state['anchorbox_scales']
        lambda_no_obj = state['lambda_no_obj']
        lambda_coord = state['lambda_coord']
        num_gtbs_per_img = state['num_gtbs_per_img']
        lamda_obj=state['lamda_obj']
        lamda_cls=state['lamda_cls']
        objectness_threshold=state['objectness_threshold']
        default_box_values_for_first_n_mb=state['default_box_values_for_first_n_mb']
        default_box_scale=state['default_box_scale']

        return TrainFunction(inputs[0], inputs[1],
                             grid_size_hor=grid_size_hor, grid_size_ver=grid_size_ver,
                             num_anchorboxes=num_anchorboxes, anchorbox_scales=anchorbox_scales,
                             num_gtbs_per_img=num_gtbs_per_img,
                             lamda_coord=lambda_coord,
                             lambda_obj=lamda_obj,
                             lamda_no_obj=lambda_no_obj,
                             lamda_cls=lamda_cls,
                             objectness_threshold=objectness_threshold,
                             default_box_values_for_first_n_mb=default_box_values_for_first_n_mb,
                             default_box_scale=default_box_scale,
                             name=name)

    ######## Setter ############

    def set_lambda_coord(self, value):
        self.lambda_coord = value

    def set_lambda_no_obj(self, value):
        assert value <= 1, "lambda_no_obj must be smaller or equal 1"
        self.lambda_no_obj = value

    ####### user functions ##########
    def create_outputs_like_cyolo(self, eval_results, gtb_inputs):


        target = np.zeros(eval_results.shape)
        scale = np.zeros(eval_results.shape)

        if(self.mb_count < self.default_box_values_for_first_n_mb):
            for y in range(self.grid_size_ver):
                for x in range(self.grid_size_hor):
                    for z in range(self.num_anchorboxes):
                        vector_index = (y*self.grid_size_hor + x)*self.num_anchorboxes + z
                        target[:,vector_index,0] += (x + .5)/self.grid_size_hor
                        target[:, vector_index, 1] += (y + .5) / self.grid_size_ver
                        target[:,vector_index,2] += self.anchorbox_scales[z][0]
                        target[:, vector_index, 3] += self.anchorbox_scales[z][1]
            scale[:,:,0:4] = self.default_box_scale
            self.mb_count+=1

        ### set classes and no_obj ###
        for sample in range(len(eval_results)):
            gtb_array = gtb_inputs[sample].reshape((int(len(gtb_inputs[sample]) / 5), 5))
            pred_bb = eval_results[sample][:, 0:4]
            pred_bb_transposed = np.transpose(pred_bb, (1, 0))
            for gtb_index in range(len(gtb_array)):
                if gtb_array[gtb_index][4] == 0: break
                """
                for y in range(self.grid_size_ver):
                    for x in range (self.grid_size_hor):
                        for z in range(self.num_anchorboxes):
                            vector_index = (y * self.grid_size_hor + x) * self.num_anchorboxes + z
                            iou = self.iou(eval_results[sample][vector_index][0:4], gtb_array[gtb_index,0:4])
                            if array_target_obj[sample][vector_index] < iou:
                                array_target_obj[sample][vector_index] = iou

                            array_goal_cls[sample][vector_index][int(gtb_array[gtb_index,4]-1)] += iou
                """
                ious = self.numpy_iou(pred_bb_transposed, gtb_array[gtb_index, 0:4])
                ious.shape += (1,)
                target[sample][:,4:5] = np.maximum(target[sample][:,4:5], ious) #objectness is not learned here! but we need to find the highest iou amogst the gtb to determine whether it is no_obj!
                gtb_cls_nr = int(gtb_array[gtb_index, 4] - 1)+5
                target[sample, :, gtb_cls_nr:gtb_cls_nr + 1] += np.select([ious>0],[1],0)

        #set no_obj
        target[:,:,4:5]=np.select(target[:,:,4:5] > self.objectness_threshold, target[:,:,4:5], 0)
        scale[:,:,4:5]=np.select([target[:,:,4:5]==0], [self.lambda_no_obj],0)

        divisor_wzero = np.add.reduce(target[:,:,5:], axis=2)
        divisor = np.zeros(divisor_wzero.shape) + divisor_wzero
        divisor[np.where(divisor == 0)] = 1

        divisor.shape += (1,)
        divisor_wzero.shape += (1,)

        target[:, :, 5:] /= divisor
        active = np.zeros(divisor.shape)
        active[np.where(divisor_wzero > 0)] = 1

        scale[:,:,5:] += active


        ### set x,y,w,h,o for the resposible box ###
        # get dimensions
        mb_size = eval_results.shape[0]
        for sample in range(mb_size):

            gtb_array = gtb_inputs[sample].reshape((int(len(gtb_inputs[sample]) / 5), 5))  # shape (50,5)
            for i in range(len(gtb_array)):
                if gtb_array[i][4] == 0: break  # gtb list is in the padding area! (cls cannot be 0!)

                # gridcell of the responsible box
                x = min(int(gtb_array[i][0] * self.grid_size_hor), self.grid_size_hor - 1)
                y = min(int(gtb_array[i][1] * self.grid_size_ver), self.grid_size_ver - 1)

                # Find ab with highest iou with gtb
                highest_iou_index = None
                highest_iou = 0

                for z in range(self.num_anchorboxes):
                    iou = self.iou((0,0,gtb_array[i][2],gtb_array[i][3]),
                                   (0,0,self.anchorbox_scales[z][0],self.anchorbox_scales[z][1]))

                    if iou > highest_iou:
                        highest_iou = iou
                        highest_iou_index = z

                # if that ab exists: set goal and scale on x,y,w,h,obj
                if (highest_iou_index is not None):
                    # self.logHit(x,y,highest_iou_index)
                    vector_index = y * self.grid_size_hor * self.num_anchorboxes + x * self.num_anchorboxes + highest_iou_index

                    # BUT set only if the gridcell is not already responsible for another gtb with higher iou
                    if highest_iou > target[sample][vector_index][4]:
                        target[sample][vector_index][0] = gtb_array[i][0]
                        target[sample][vector_index][1] = gtb_array[i][1]
                        target[sample][vector_index][2] = gtb_array[i][2]
                        target[sample][vector_index][3] = gtb_array[i][3]
                        target[sample][vector_index][4] = highest_iou

                        scale[sample][vector_index][0:4] = self.lambda_coord
                        scale[sample][vector_index][4] = self.lamda_obj * (2 - gtb_array[i][2]*gtb_array[i][3])#TODO check if 2-... is necessary

        return target, scale

    @staticmethod
    def iou(box1, box2):
        leftbound =  max(box1[0]-.5*box1[2], box2[0]-.5*box2[2])
        rightbound = min(box1[0] + .5 * box1[2], box2[0] + .5 * box2[2])

        lowerbound = max(box1[1] - .5 * box1[3], box2[1] - .5 * box2[3])
        upperbound = min(box1[1] + .5 * box1[3], box2[1] + .5 * box2[3])

        x_inter = max(rightbound - leftbound,0)
        y_inter = max(upperbound - lowerbound,0)

        intersection = x_inter * y_inter
        area1 = box1[2]*box1[3]
        area2 = box2[2]*box2[3]
        union = area1 + area2 - intersection

        return intersection / union

    @staticmethod
    def numpy_iou(boxes1, boxes2):
        """boxes1 and boxes2 in shape of (4,n)"""
        leftbound =  np.maximum(boxes1[0]-.5*boxes1[2], boxes2[0]-.5*boxes2[2])
        rightbound = np.minimum(boxes1[0] + .5 * boxes1[2], boxes2[0] + .5 * boxes2[2])

        lowerbound = np.maximum(boxes1[1] - .5 * boxes1[3], boxes2[1] - .5 * boxes2[3])
        upperbound = np.minimum(boxes1[1] + .5 * boxes1[3], boxes2[1] + .5 * boxes2[3])

        x_inter = np.maximum(rightbound - leftbound,0)
        y_inter = np.maximum(upperbound - lowerbound,0)

        intersection = x_inter * y_inter
        area1 = boxes1[2]*boxes1[3]
        area2 = boxes2[2]*boxes2[3]
        union = area1 + area2 - intersection

        return intersection / union



    ######## applies sqrt() to w & h of the volume; uses cntk ops ######### (YOLOv1 only?)

    @staticmethod
    def make_wh_sqrt3(network):
        xy = network[:,:,0:2]
        wh = network[:,:,2:4]
        rest=network[:,:,4:]

        sqrt_wh = sqrt(wh)
        spliced = splice(xy,sqrt_wh,rest,axis=2)

        return spliced

    @staticmethod
    def make_wh_sqrt(network):
        xy = network[:,0:2]
        wh = network[:,2:4]
        rest=network[:,4:]

        sqrt_wh = sqrt(wh)
        spliced = splice(xy,sqrt_wh,rest,axis=1)

        return spliced

    ######### Check values in numpy ##########

    def sqrt_np_wh(self, volume):
        xy = volume[:, 0:2]
        wh = volume[:, 2:4]
        rest = volume[:, 4:]

        sqrt_wh = np.sqrt(wh)
        ap1 = np.append(xy, sqrt_wh, axis=1)
        ap2 = np.append(ap1, rest, axis=1)
        return ap2  # np.append(xy, sqrt_wh, rest, axis=1)

    # this is a test-function for development and may be disabled during actual training
    def check_values(self, eval_values, target, scale):
        if (np.isnan(eval_values).any()):
            print("Model output contained nan!")
            print(eval_values.shape)
        elif np.greater(eval_values, 100).any():
            print("greater 100!")
            print(np.max(eval_values))

            # import ipdb
            # ipdb.set_trace()

        if (np.isnan(target).any()):
            print("error_volume contained nan!")

        if (np.equal(target[:, :, 4:5], 1).any()):
            print("objectness target == 1! NOT ALLOWED!")

        if ((np.greater(scale[:, :, 2:4], 0) & np.equal(target[:, :, 2:4], 0)).any()):
            # if(np.where(scale[ np.where(np.equal(target[:,:,2:4], 0))]>0)[0].any() ):
            print("Scale is > 0 where target == 0 --> NOT ALLOWED")

        if (np.isnan(scale).any()):
            print("scale_volume contained nan!")

        # err = TrainFunction2.sqrt_np_wh(self,eval_values) - TrainFunction2.sqrt_np_wh(self,error)  # substrac "goal" --> error
        err = eval_values - target
        sq_err = err * err
        sc_err = sq_err * scale  # apply scales (lambda_coord, lambda_no_obj, zeros on not learned params)
        mse = np.add.reduce(sc_err, axis=None)
        if (math.isnan(float(mse))):
            print("mse is nan!")
            import ipdb
            ipdb.set_trace()
            exit()