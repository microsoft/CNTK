# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from tools.DAG import *
from cntk.ops.functions import *
import numpy as np

# def _score_validity(node):
#     """
#     Internal function tagging a node as valid or invalid. Used for reduction of the graph.
#     :param node: graph node to be tested
#     :return: boolean determining if the node is valid
#     """
#     node.valid = node.next or node.strings
#     return node.valid
#
#
# def _remove_valid_scoring(node):
#     """
#     Internal function which removes the tagging applied by _score_validity(node).
#     :param node: graph node to be untagged
#     :return: None
#     """
#     node.valid = None
#     for i in range(len(node.next)):
#         _remove_valid_scoring(node.next[i])
#
#
# def DAG_to_Tree(DAG_root):
#     """
#     Turns a DAG to a tree by deleting connections as required. The DAG must have a single root/start node.
#     :param DAG_root: root/start node of the DAG
#     :return: param DAG_root
#     """
#     assert not hasattr(DAG_root,
#                        "valid"), "Node's validity may not be scored in a DAG. Use remove_valid_scoring() before!"
#     # kill the connection to every valid child
#     keep = []
#     for i in range(len(DAG_root.next)):
#         child = DAG_root.next[i]
#         # If a child is already valid (has another parent), remove it
#         if hasattr(child, "valid"):
#             child.prev.remove(DAG_root)
#         else:
#             keep.append(child)
#             DAG_to_Tree(child)
#
#     DAG_root.next = keep
#     _score_validity(DAG_root)
#
#     return DAG_root
#
# def remove_invalid_leafs_from_Tree(node):
#     """
#     Removes leafs which do not contain a string from the tree.
#     :param node: node whose childrens are to be checked
#     :return: None
#     """
#     _score_validity(node)
#     if not node.valid:
#         for i in range(len(node.prev)):
#             parent = node.prev[i]
#             parent.next.remove(node)
#             remove_invalid_leafs_from_Tree(parent)
#         node.prev.clear()
#     else:
#         for i in reversed(range(len(node.next))):
#             remove_invalid_leafs_from_Tree(node.next[i])
#
# def remove_nodes_with_one_child_only(node):
#     """
#     Removes nodes from a tree which have only one child and no string.
#     :param node: node which should be checked recursively downwards.
#     :return: the node replacing this node - may be param node if it need not be removed
#     """
#     LONELY_NODES_ONLY = False
#
#     if len(node.next)==1 and not node.strings:
#         only_child = node.next[0]
#         if node.prev:
#             for i in range(len(node.prev)):
#                 parent = node.prev[i]
#                 if LONELY_NODES_ONLY and len(parent.next) > 1:
#                     remove_nodes_with_one_child_only(only_child)
#                     return node
#                 parent.next[parent.next.index(node)]=only_child
#             only_child.prev.remove(node)
#             only_child.prev.extend(node.prev)
#         else:
#             #node is discarded and is the root node!
#             only_child.prev.remove(node)
#         return remove_nodes_with_one_child_only(only_child)
#     else:
#         for i in range (len(node.next)):
#             remove_nodes_with_one_child_only(node.next[i])
#         return node

class TreeMap():
    @staticmethod
    def tree_map_from_tree_str(tree_str, use_background=True, use_multiply_with_parent=True):
        """
        Creates a TreeMap by deserealizing from a string.
        :param tree_str: str to be deserialized from
        :param use_background: whether or not a background class should be used
        :param use_multiply_with_parent: whether or not the outputs each classes conditional propability shall be multiplied with its parents propability.
        :return: the deserialized TreeMap
        """
        def _find_nodes_of_cls_map(node, cls_map):
            nodes = []
            tagged = []
            todo=[node]

            while todo:
                lnode = todo.pop()
                if lnode in tagged: continue
                if cls_map in lnode.cls_maps:
                    nodes.append(lnode)
                tagged.append(lnode)
                todo.extend(lnode.next)
                todo.extend(lnode.prev)

            return nodes

        root_node, cls_maps = DAG_Utils.deserialize(tree_str)
        meta_map={}
        for cls_map in cls_maps:
            nodes = _find_nodes_of_cls_map(root_node, cls_map)
            node_map = {}

            for name in cls_map.cls_map:
                for node in nodes:
                    if name in node.strings:
                        node_map[name]=node
                        break
            meta_map[cls_map] = node_map

        return TreeMap(root_node, meta_map, use_background=use_background, use_multiply_with_parent=use_multiply_with_parent)


    def __init__(self, root_node, meta_map,use_background=True, use_multiply_with_parent=True):
        self.root_node = root_node
        self.meta_map = meta_map
        self.use_background = use_background
        self.use_multiply_with_parent = use_multiply_with_parent
        self._create_nr_of_entries()

    def refresh_tree(self):
        self._create_single_row_view()
        self._create_nr_of_entries()

    def _create_single_row_view(self):
        to_do = [self.root_node]
        out=[[self.root_node]]
        completely_flat = [self.root_node]
        while to_do:
            node = to_do[0]
            to_do = to_do[1:]
            if node.next:
                out.append(node.next)
                to_do.extend(node.next)
                if self.use_background: completely_flat.append(None)
                completely_flat.extend(node.next)

        self.single_row_view = out
        self.as_in_network = completely_flat[1:] # all nodes behind another, excluding root, including backgrouds as None
        self._create_index_tagging_in_tree()
        regions = []
        for i in range(1,len(out)):
            part = out[i]
            min_index = part[0]._index_in_network - (1 if self.use_background else 0)
            max_index = part[-1]._index_in_network
            regions.append((min_index, max_index+1))
        self.softmax_regions = regions # ignore root

    def _create_index_tagging_in_tree(self):
        for i in range(len(self.as_in_network)):
            node = self.as_in_network[i]
            if node is not None: node._index_in_network = i

    def _create_nr_of_entries(self):
        if not hasattr(self, "single_row_view"): self._create_single_row_view()

        count=0
        for i in range(len(self.single_row_view)):
            count+=len(self.single_row_view[i])
        self.nr_of_entries = count

    def get_nr_of_required_neurons(self):
        """
        Returns the number of neurons required perpredictor with this TreeMap
        :return: int number of required neurons
        """
        return len(self.as_in_network)

    def get_sofmax_regions_length(self):
        if not hasattr(self, "single_row_view"): self._create_single_row_view()

        out = []
        for i in range(len(self.single_row_view-1)):
            out.append(len(self.single_row_view[i+1]) + (1 if self.use_background else 0)) # +1 to ignore root!
        return out

    def tree_softmax(self, model, axis=0, offset=0):
        """
        Applies the softmax for the hierarchical prediction.
        :param model: the model the prediction shall be applied to
        :param axis: the axis along which the prediction shall be made
        :param offset: nr of neurons not to be assigned to (on axis, beginning with 0). Can be shift the predictor away from eg. bounging box predictions.
        :return: function with model as input
        """
        def _multiply_with_parents(slices, tree_map):
            regions = tree_map.softmax_regions # first region is below the root node and need not be multiplied therefore!
            my_input_slices = slices[(0 if offset == 0 else 1):] #synchronise to regions
            my_output_slices = slices[0:(1 if offset == 0 else 2)]#extract those w/o need to be multiplied
            current_parents=slices[(0 if offset == 0 else 1)]

            for i in range(1, len(regions)): #skip region below root since it need not be multiplied
                first_node = tree_map.as_in_network[regions[i][0]] # first node in region to be multiplied
                if first_node is None: first_node = tree_map.as_in_network[regions[i][0]+1] #when using background, the first node is set to none (background)
                parent = first_node.prev[0] #it  must be a tree so no aother parents are valid now!
                parent_index_in_network=parent._index_in_network

                #find responsible region
                j =0
                while j<len(regions) and not (regions[j][0]<=parent_index_in_network<regions[j][1]):
                    j+=1
                offset_in_region = parent_index_in_network - regions[j][0]

                multiplier = cntk.ops.slice(my_output_slices[j], axis=axis, begin_index=offset_in_region, end_index=offset_in_region+1, name="parent_value_from_region_"+str(j)+"_offset_"+str(offset_in_region))
                multiplier = cntk.ops.stop_gradient(multiplier)
                out = multiplier * my_input_slices[i]

                my_output_slices.append(out)


            return my_output_slices

        slices = [] if offset == 0 else [cntk.ops.slice(model, axis=axis, begin_index=0, end_index=offset)]
        for i in range(len(self.softmax_regions)):
            indices = self.softmax_regions[i]
            region = cntk.ops.slice(model, axis=axis, begin_index=indices[0]+offset, end_index=indices[1]+offset, name="Region_"+str(i)+"_from_"+str(indices[0]+offset)+"_to_"+str(indices[1]+offset))
            softmaxed = cntk.ops.softmax(region, axis=axis)
            slices.append(softmaxed)
        if(self.use_multiply_with_parent):
            slices = _multiply_with_parents(slices, self)
        print("applied " + str(i) + " softmax_regions")
        return cntk.ops.splice(*slices, axis=axis)

    def get_train_softmax_vectors(self, to_Set, scale_value, dtype=np.float32): # to_set: List<Tuple<ClassMap, int>>
        """
        Creates target and scale vector for a given prediction.
        :param to_Set: list of tuples (cls_map, original_label) of labels to predict. The original_label refers to the label in the given ClassMap.
        :param scale_value: value for the scale vector on active regions
        :param dtype: datatype for the returned vectors
        :return: tuple of vectors (numpy array) (target, scale)
        """
        if not hasattr(self, "nr_of_entries"): self._create_nr_of_entries()

        target = np.zeros(self.get_nr_of_required_neurons(), dtype=dtype)
        scale = np.zeros(target.shape, dtype=dtype)

        for i in range(len(to_Set)):
            cls_map = to_Set[i][0]
            value = to_Set[i][1]
            if value == 0:
                continue
            node = self.meta_map[cls_map][cls_map.getClass(value)]
            while node.prev:
                if not hasattr(node, "_index_in_network"): import ipdb;ipdb.set_trace()
                target[node._index_in_network]+=1
                node = node.prev[0]


        for i in range(len(self.softmax_regions)):
            local_range = self.softmax_regions[i]
            hits = np.add.reduce(target[local_range[0]:local_range[1]])
            if hits>0:
                target[local_range[0]:local_range[1]]/=hits
                scale[local_range[0]:local_range[1]]=scale_value
            elif hits == 0 and self.use_background:
                target[local_range[0]]=1 # set backgound to 1
                scale[local_range[0]:local_range[1]] = scale_value

        return target, scale

    def get_output_mapper(self):
        """
        Getter for an Output_Mapper for this TreeMap
        :return: Output_Mapper for this TreeMap
        """
        return Output_Mapper(self)

class Output_Mapper():
    all_classes = []

    def __init__(self, tree_map):
        self.use_background = tree_map.use_background
        self.input_is_absolut = tree_map.use_multiply_with_parent

        in_net_array = np.asarray(tree_map.as_in_network)
        self.keep_indicies=np.where( np.not_equal( in_net_array[1:],[None]))
        tmp = self.keep_indicies[0]
        tmp+=1
        self.all_classes=[]
        if self.use_background:
            self.all_classes.append("__background__")
        entries=in_net_array[self.keep_indicies]
        for i in range(len(entries)):
            if entries[i].strings:
                self.all_classes.append(entries[i].strings[0])
            else:
                self.all_classes.append("unnamed entity")

        if self.use_background:
            self.keep_indicies = (np.append([0], self.keep_indicies[0]),)

    def get_all_classes(self):
        """
        Getter for all known classes in this mapper
        :return: list of str with each known classname
        """
        return self.all_classes.copy()

    def get_prediciton_vector(self, network_output):
        """
        Removes the background predictions of the different softmax-regions despite the root one if use_background is true. Thereby mapping outputs to the indicies in the extended classmap.
        :param network_output: the vector the network predicted
        :return: the predictions without the background fields
        """
        return network_output[self.keep_indicies]
