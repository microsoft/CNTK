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

class TrainFunction2(UserFunction):


    def __init__(self,
                 arg1, gtbs,
                 grid_size_hor=(par_image_width/par_downsample),
                 grid_size_ver=(par_image_height / par_downsample),
                 num_anchorboxes=par_num_anchorboxes,
                 anchorbox_scales=par_anchorbox_scales,
                 num_gtbs_per_mb=par_max_gtbs,
                 lambda_coord =5.0*500,
                 lambda_no_obj = 0.5,
                 name="TrainFunction"):
        super(TrainFunction2, self).__init__([arg1, gtbs], name=name)
        self.grid_size_hor = int(grid_size_hor)
        self.grid_size_ver = int(grid_size_ver)
        self.num_anchorboxes = int(num_anchorboxes)
        self.lambda_no_obj = lambda_no_obj
        self.lambda_coord = lambda_coord
        self.anchorbox_scales  = anchorbox_scales
        self.num_gtbs_per_mb = int(num_gtbs_per_mb)

    def set_lambda_coord(self, value):
        self.lambda_coord = value

    def forward(self, arguments, outputs=None, keep_for_backward=None, device=None, as_numpy=False):
        roi_vectors_list, scale_vectors_list = self.create_outputs(arguments[0], arguments[1])

        outputs[self.outputs[0]] = np.ascontiguousarray(roi_vectors_list, np.float32)
        outputs[self.outputs[1]] = np.ascontiguousarray(scale_vectors_list, np.float32)

        if False:
            self.check_values(arguments[0], outputs[self.outputs[0]], outputs[self.outputs[1]])

        return None

    def sqrt_np_wh(self, volume):
        xy = volume[:, 0:2]
        wh = volume[:, 2:4]
        rest = volume[:, 4:]

        sqrt_wh = np.sqrt(wh)
        ap1 = np.append(xy, sqrt_wh, axis=1)
        ap2= np.append(ap1, rest, axis =1 )
        return ap2 #np.append(xy, sqrt_wh, rest, axis=1)

    def check_values(self, eval_values, target, scale):
        if(np.isnan(eval_values).any()):
            print("Model output contained nan!")
            print(eval_values.shape)
        elif np.greater(eval_values, 100).any():
            print("greater 100!")
            print(np.max(eval_values))

            #import ipdb
            #ipdb.set_trace()

        if(np.isnan(target).any()):
            print("error_volume contained nan!")

        if(np.equal(target[:,:,4:5],1).any()):
            print("objectness target == 1! NOT ALLOWED!")

        if((np.greater(scale[:,:,2:4],0) & np.equal(target[:,:,2:4],0)).any()):
        #if(np.where(scale[ np.where(np.equal(target[:,:,2:4], 0))]>0)[0].any() ):
            print("Scale is > 0 where target == 0 --> NOT ALLOWED")

        if(np.isnan(scale).any()):
            print("scale_volume contained nan!")


        #err = TrainFunction2.sqrt_np_wh(self,eval_values) - TrainFunction2.sqrt_np_wh(self,error)  # substrac "goal" --> error
        err = eval_values - target
        sq_err = err * err
        sc_err = sq_err * scale  # apply scales (lambda_coord, lambda_no_obj, zeros on not learned params)
        mse = np.add.reduce(sc_err, axis=None)
        if (math.isnan(float(mse))):
            print("mse is nan!")
            import ipdb
            ipdb.set_trace()
            exit()

    def backward(self, state, root_gradients, variables):
        return None


    def infer_outputs(self):
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes, needs_gradient=False),
                output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes, needs_gradient=False)]

    def serialize(self):
        internal_state = {}
        internal_state['grid_size_hor'] = self.grid_size_hor
        internal_state['grid_size_ver'] = self.grid_size_ver
        internal_state['num_anchorboxes'] = self.num_anchorboxes
        internal_state['anchorbox_scales'] = self.anchorbox_scales
        internal_state['lambda_no_obj'] = self.lambda_no_obj
        internal_state['lambda_coord'] = self.lambda_coord
        internal_state['num_gtbs_per_mb'] = self.num_gtbs_per_mb
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
        num_gtbs_per_mb = state['num_gtbs_per_mb']
        return TrainFunction2(inputs[0], inputs[1],
                             grid_size_hor=grid_size_hor, grid_size_ver=grid_size_ver,
                             num_anchorboxes=num_anchorboxes, anchorbox_scales=anchorbox_scales,
                             num_gtbs_per_mb=num_gtbs_per_mb,
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
        array_goal = np.zeros(eval_results.shape, np.float32)
        array_scale = np.asarray(list_scales, np.float32)
        for slice in range(mb_size):


            gtb_array = gtb_inputs[slice].reshape((int(len(gtb_inputs[slice])/5) , 5)) #shape (50,5)
            # Here we are going completely for numpy and for loops and no cntk/parallised ops! TODO Switch to CNTK implementation
            for i in range(len(gtb_array)):
                if gtb_array[i][4] == 0 : break # gtb list is in the padding area!
                #find gridcells who are inside of the gtb
                left_gc_index = int((gtb_array[i][0] - 0.5 * gtb_array[i][2]) * self.grid_size_hor) #use floor to cast to int
                left_gc_index = max(left_gc_index, 0) # use clip in ctk to set bounds
                right_gc_index =int((gtb_array[i][0] + 0.5 * gtb_array[i][2]) * self.grid_size_hor)
                right_gc_index = min(right_gc_index, self.grid_size_hor)

                top_gc_index =  int((gtb_array[i][1] - 0.5 * gtb_array[i][3]) * self.grid_size_ver)
                top_gc_index = max(top_gc_index, 0)
                bottom_gc_index = int((gtb_array[i][1] + 0.5 * gtb_array[i][3]) * self.grid_size_ver)
                bottom_gc_index = min(bottom_gc_index, self.grid_size_ver)

                # begin  new
                gc_x_index =max( int(gtb_array[i][0] * self.grid_size_ver), self.grid_size_ver-1)
                gc_y_index =max( int(gtb_array[i][1] * self.grid_size_hor), self.grid_size_hor-1)
                if True:
                    if True:
                        x= gc_x_index
                        y= gc_y_index

                #for x in range(left_gc_index, right_gc_index + 1): # create new tensor volume for each gtb, reduce_sum to bring it back together!
                #    for y in range (top_gc_index, bottom_gc_index +1 ):
                # end new
                        highest_iou_index = None
                        highest_iou = 0
                        for z in range (self.num_anchorboxes):
                            curr_vector = y * self.grid_size_ver * self.num_anchorboxes + x * self.num_anchorboxes + z
                            #score iou to find bb
                            ab_iou = self.calc_iou((gtb_array[i][0],gtb_array[i][1],gtb_array[i][2],gtb_array[i][3]),
                                                   ((x+0.5) * 1.0/self.grid_size_hor, (y+0.5) * 1.0/self.grid_size_ver, self.anchorbox_scales[z][0], self.anchorbox_scales[z][1]))

                            # Set Pr(Class|Obj) for the class found to 1
                            # if a gc has multiple classes ALL are set!
                            if(array_goal[slice][curr_vector][4] < ab_iou):
                                array_goal[slice][curr_vector][5 + int(gtb_array[i][4]) - 1] = 1
                                for cls_sc_i in range(num_cls):
                                    array_goal[slice][curr_vector][5 + cls_sc_i] = 1 if int(gtb_array[i][4] - 1) == cls_sc_i else 0
                                    array_scale[slice][curr_vector][5 + cls_sc_i] = 1

                                # save iou to determine whether ab contained the obj or noobj constant needs to be applied!
                                array_goal[slice][y * self.grid_size_ver * self.num_anchorboxes + x * self.num_anchorboxes + z][4] = ab_iou

                            if ab_iou > highest_iou:
                                highest_iou = ab_iou
                                highest_iou_index = z

                        vector_index = y * self.grid_size_ver * self.num_anchorboxes + x * self.num_anchorboxes + highest_iou_index if highest_iou_index is not None else 0
                        if(highest_iou_index is not None and highest_iou > array_goal[slice][vector_index][4]):
                            # if an ab is in scope and no other gtb with higher iou has jet trained this ab
                            array_goal[slice][vector_index][0] = gtb_array[i][0]
                            array_goal[slice][vector_index][1] = gtb_array[i][1]
                            array_goal[slice][vector_index][2] = gtb_array[i][2]
                            array_goal[slice][vector_index][3] = gtb_array[i][3]
                            #array_goal[slice][vector_index][4] = highest_iou

                            for z in range(4):
                                array_scale[slice][vector_index][z] = self.lambda_coord


                            if False:
                                for z in range(self.num_anchorboxes):
                                    curr_index = y * self.grid_size_ver * self.num_anchorboxes + x * self.num_anchorboxes + z
                                    if(array_goal[slice][curr_index][4] > 0):
                                        # TODO scale is set to 0 for other bounding boxes containing the gtb. In the paper it is unclear whether scale should be 0 or lambda_no_obj and goal= 0 (or = iou?)
                                        array_scale[slice][curr_index] = 1# if curr_index == vector_index else 0

        #   default: set scale(x,y,w,h,cls)=0, set scale(obj)=lambda_no_obj, set goal(obj)=0 (set goal(*)=0)
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

        if True:
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

            left, right, top, bottom = None, None, None, None
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

        else:
            x1 = coords1[0]
            x2 = coords2[0]
            y1 = coords1[1]
            y2 = coords2[1]
            w1 = coords1[2]
            w2 = coords2[2]
            h1 = coords1[3]
            h2 = coords2[3]


            dist_x = math.abs(x1 - x2)
            dist_y = math.abs(y1 - y2)

            intersec_x = (w1 + w2) - dist_x
            intersec_y = (h1 + h2) - dist_y

            intersection = max(0, intersec_x) * max(0,intersec_y)

        assert intersection >= 0, "Intersection must be equal or greater 0"

        return intersection / (area1 + area2 - intersection)

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

    def create_outputs_cntk(self, eval_results, gtb_inputs):
        gtb_arrays = reshape(gtb_inputs, (gtb_inputs.shape[0], self.num_gtbs_per_mb, 5))
        cls_vectors = one_hot(gtb_arrays[:,:,0:1], par_num_classes) # TODO make independent from PARAM to remain ability to deserialize correctly
        gtb_positions = gtb_arrays[:,:,1:] # cut off class!



        #   defaulft= set scale(x,y,w,h,cls)=0, set scale(obj)=lambda_no_obj, set goal(obj)=0 (set goal(*)=0)
        #   find gridcells who are inside of the gtb
        #      batch?: set the cls scale to 1, set goal to 0default and 1 where class
        #              set scale(obj) = 1, goal(obj)=IOU (might have to be done single_op)
        #      single_op: find anchorboxes which should be trained
        #         set x,y,w,h of those (goal=gtb, scale = lambda_cord)
        #         set objness of those goal=1, scale =1
        #


    @staticmethod
    def multi_iou(coords1, coords2):
        '''
        Calulates the IOU of the boxes specified in coords1 and coords2.
        :param coords1: Must have the shape (number_of_boxes, box_coordinates), where number_of boxes of coords1 and coords2 must be equal. box_xoordinates are specified as [X, Y, W, H], where X and Y describe the X- and Y-Coordinate of the boxes center and W and H describe the boxes width and height.
        :param coords2: See coord1
        :return: Output of shape (number_of_boxes, 1) with the element at index i specified by the IOU of the boxes described by coords1[i] and coords2[i].
        '''
        w1 = coords1[:,2:3]
        w2 = coords2[:,2:3]
        h1 = coords1[:,3:4]
        h2 = coords2[:,3:4]

        x1 = coords1[:,0:1]
        x2 = coords2[:,0:1]
        y1 = coords1[:,1:2]
        y2 = coords2[:,1:2]

        max_intersex_x = element_min(w1,w2)
        max_intersex_y = element_min(h1,h2)
        dist_x = abs(x1 - x2)
        dist_y = abs(y1 - y2)
        unconstrained_intersec_x = (w1 + w2) / 2  - dist_x
        unconstrained_intersec_y = (h1 + h2) / 2 - dist_y
        intersec_x = element_min(unconstrained_intersec_x, max_intersex_x) # set maximum bounds in case one box is completely inside of the other
        intersec_y = element_min(unconstrained_intersec_y, max_intersex_y)
        intersection = relu(intersec_x) * relu(intersec_y)

        area1 = w1 * h1
        area2 = w2 * h2
        union = (area1 + area2) - intersection

        return intersection / union
        #TODO check against IOU-impl from faster RCNN