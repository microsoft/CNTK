# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
#
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
from cntk.ops.functions import UserFunction
from cntk import *

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
from tools.Tree_Creator import TreeMap


class HierarchyHelper:

    def __init__(self, tree_str):
        # Constants
        self.MINIMUM_BG_VALUE = .65 # Constant which determines the minimum background value for a Detection to be background
        self.tree_map = TreeMap.tree_map_from_tree_str(tree_str, use_background=True, use_multiply_with_parent=False)
        self.cls_maps = list(self.tree_map.meta_map.keys())

        self.output_mapper = self.tree_map.get_output_mapper()
        
    def get_vectors_for_label_nr(self, label):
        """
        Creates target and scale vector for the given label. Requires that only one classmap is used!
        :param label: label as in the ClassMap (int)
        :return: tuple of vectors (target, scale)
        """
        return self.tree_map.get_train_softmax_vectors(to_Set=[(self.cls_maps[0], label)], scale_value=1)

    def get_vectors_for_label(self, label):
        """
        Creates target and scale vector for the given label. Requires that only one classmap is used!
        :param label: label as one-hot vector
        :return: tuple of vectors (target, scale)
        """
        index = np.argmax(label)
        return self.get_vectors_for_label_nr(index)

    def top_down_eval(self, vector):
        """
        Performs a top-down evaluation of the given predicted vector.
        :param vector: prediction as by the network
        :return: Vector of the same shape as the input vector, where the predicted classes are assigned their likeliness and 0 for all other classes.
        """
        out_vec = np.zeros(vector.shape, dtype=np.float32)

        start = 0
        multiplier = 1

        for region in self.tree_map.softmax_regions:
            if start >= region[1]: continue

            if self.tree_map.use_background:
                if vector[region[0]] < self.MINIMUM_BG_VALUE:
                    node_index = np.argmax(vector[region[0]+1:region[1]], axis=0) + region[0]+1
                else:
                    node_index = np.argmax(vector[region[0]: region[1]], axis=0) + region[0]
            else:
                node_index = np.argmax(vector[region[0]:region[1]], axis=0) + region[0]
            #if np.add.reduce(out_vec)==0 and np.argmax(vector[0:3])!=0: import ipdb;ipdb.set_trace()
            node = self.tree_map.as_in_network[node_index]
            if node is None: # background --> stop here!
                if node_index == 0:# first bg --> set global bg
                    out_vec[0] = vector[0]
                    assert (out_vec[0]!=0 or vector[0]==0)
                break

            if self.tree_map.use_background:
                bg_value = vector[region[0]]
                bg_removal_factor = 1 / (1 - bg_value)
                multiplier *= bg_removal_factor

            multiplier *= vector[node_index]
            out_vec[node_index] = multiplier

            if not node.next: # leaf
                assert np.add.reduce(out_vec) > 0
                break


            # else
            start = node.next[0]._index_in_network

        if np.add.reduce(out_vec)==0: import ipdb;ipdb.set_trace()
        return out_vec

    def apply_softmax(self, model, axis=0, offset=0):
        """
        Applies the softmax for the hierarchical prediction.
        :param model: the model the prediction shall be applied to
        :param axis: the axis along which the prediction shall be made
        :param offset: nr of neurons not to be assigned to (on axis, beginning with 0). Can be shift the predictor away from eg. bounging box predictions.
        :return: function with model as input
        """
        return self.tree_map.tree_softmax(model, axis=axis, offset=offset)

class Target_Creator(UserFunction):
    """
    Takes standard one_hot input according to the datasets classmap and turns it into target vector. Therefore is a label-converter.
    """
    def __init__(self,
            arg,
            max_nr_of_rois,
            hierarchy_helper,
            name='',):

        super(Target_Creator, self).__init__([arg], name=name)
        self.max_nr_of_rois = max_nr_of_rois
        self.hhelper=hierarchy_helper

    def infer_outputs(self):
        return [output_variable((self.max_nr_of_rois, self.hhelper.tree_map.get_nr_of_required_neurons()), self.inputs[0].dtype, self.inputs[0].dynamic_axes, needs_gradient=False)]

    def forward(self, argument, device=None, outputs_to_retain=None):
        mb_size= len(argument)
        nrRois= len(argument[0])
        output = np.zeros((mb_size, nrRois, self.hhelper.tree_map.get_nr_of_required_neurons() ), dtype=np.float32)
        for i in range(mb_size):
            for j in range(nrRois):
                target,_=self.hhelper.get_vectors_for_label(argument[i][j])
                output[i][j] = target

        return None, output

    def backward(self, state, root_gradients):
        return root_gradients

    def clone(self, cloned_inputs):
        return self.__class__(*cloned_inputs, name=self.name)
