# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from tools.hierarchy.Syn_DAC import *
import tools.hierarchy.Wordnet_Tools as wn
from cntk.ops.functions import *
import numpy as np
from tools.ClassMap import ClassMap



def create_DAC_from_clsmap(cls_map, root=None):
    if root is None: root =DAC_Utils.get_new_root_node()

    node_map = {}
    node_map["__root__"] = root
    syns_map = wn.get_top_synsets(cls_map.cls_map)
    for i in range(cls_map.getEntries()):
        string = cls_map.getClass(i)
        if string == "__background__": continue
        syn = syns_map[string]
        node = DAC_Utils.add_synset_to_DAC(root, syn, cls_map, string)
        node_map[string] = node

    return node_map


def create_DAC_from_clsmaps(cls_maps, root=None):
    mapping = {}
    for i in range(len(cls_maps)):
        cls_map = cls_maps[i]
        node_map = create_DAC_from_clsmap(cls_map, root=root)
        mapping[cls_map] = node_map
        if root is None or True: # TODO: warum einschraenken?
            root = node_map['__root__']

    return mapping


def score_validity(node):
    node.valid = node.next or node.strings
    return node.valid


def remove_valid_scoring(node):
    node.valid = None
    for i in range(len(node.next)):
        remove_valid_scoring(node.next[i])


def DAC_to_Tree(DAC_root):
    assert not hasattr(DAC_root,
                       "valid"), "Node's validity may not be scored in a DAC. Use remove_valid_scoring() before!"
    # kill the connection to every valid child
    keep = []
    for i in range(len(DAC_root.next)):
        child = DAC_root.next[i]
        # If a child is already valid (has another parent), remove it
        if hasattr(child, "valid"):
            child.prev.remove(DAC_root)
        else:
            keep.append(child)
            DAC_to_Tree(child)

    DAC_root.next = keep
    score_validity(DAC_root)

    return DAC_root

def remove_invalid_leafs_from_Tree(node):
    score_validity(node)
    if not node.valid:
        for i in range(len(node.prev)):
            parent = node.prev[i]
            parent.next.remove(node)
            remove_invalid_leafs_from_Tree(parent)
        node.prev.clear()
    else:
        for i in reversed(range(len(node.next))):
            remove_invalid_leafs_from_Tree(node.next[i])

def remove_nodes_with_one_child_only(node):
    LONELY_NODES_ONLY = False

    if len(node.next)==1 and not node.strings:
        only_child = node.next[0]
        if node.prev:
            for i in range(len(node.prev)):
                parent = node.prev[i]
                if LONELY_NODES_ONLY and len(parent.next) > 1:
                    remove_nodes_with_one_child_only(only_child)
                    return node
                parent.next[parent.next.index(node)]=only_child
            only_child.prev.remove(node)
            only_child.prev.extend(node.prev)
        else:
            #node is discarded and is the root node!
            only_child.prev.remove(node)
        return remove_nodes_with_one_child_only(only_child)
    else:
        for i in range (len(node.next)):
            remove_nodes_with_one_child_only(node.next[i])
        return node



class TreeMap():

    def __init__(self, cls_maps, use_background=True, use_multiply_with_parent=True, only_valid_leafs=False, reduce_graph=False):
        self.root_node = DAC_Utils.get_new_root_node()
        self.meta_map = create_DAC_from_clsmaps(cls_maps, self.root_node)
        DAC_to_Tree(self.root_node)
        if only_valid_leafs: remove_invalid_leafs_from_Tree(self.root_node)
        if reduce_graph: self.root_node = remove_nodes_with_one_child_only(self.root_node)
        remove_valid_scoring(self.root_node)

        self.use_background=use_background
        self.use_multiply_with_parent=use_multiply_with_parent
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
        return len(self.as_in_network)

    def get_sofmax_regions_length(self):
        if not hasattr(self, "single_row_view"): self._create_single_row_view()

        out = []
        for i in range(len(self.single_row_view-1)):
            out.append(len(self.single_row_view[i+1]) + (1 if self.use_background else 0)) # +1 to ignore root!
        return out


    #@Function
    def tree_softmax(self, model, axis=0, offset=0):
        #lenghtes = self.get_sofmax_regions_length()
        #slices = [] if offset == 0 else [slice(model, axis=axis, begin_index=0, end_index=offset)]
        #for i in range(len(lenghtes)-1):
        #    size = lenghtes[i+1] # Root does not need a softmax!
        #    region = slice(model, axis=axis, begin_index=offset,  end_index=offset+size)
        #    softmaxed = cntk.ops.softmax(region, axis=axis)
        #    slices.append(softmaxed)
        #return cntk.ops.splice(slices, axis = axis)
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
        if not hasattr(self, "nr_of_entries"): self._create_nr_of_entries()

        target = np.zeros(self.get_nr_of_required_neurons(), dtype=dtype)
        scale = np.zeros(target.shape, dtype=dtype)

        #import ipdb;ipdb.set_trace()

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

    def get_prediction(self):
        return NotImplemented





def dummy():
    class_map_files = [
        #r"D:\local\CNTK-2-0-rc1\cntk\Examples\Image\DataSets\Pascal\mappings\class_map.txt"
        r"D:\local\CNTK-2-0-rc1\cntk\Examples\Image\DataSets\Overfit\class_map.txt"
    ]
    class_maps = []
    for i in range(len(class_map_files)):
        class_maps.append(ClassMap(class_map_files[i]))
    return TreeMap(class_maps)


def create_treemap_for_cls_map(cls_map, dac_map):
    node_map = dac_map[cls_map]
