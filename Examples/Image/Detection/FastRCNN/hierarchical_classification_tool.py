
from tools.hierarchy.Tree_Creator import TreeMap
from tools.ClassMap import ClassMap
import sys, path
import PARAMETERS as par
import numpy as np
from cntk.ops.functions import UserFunction
from cntk import *

cls_maps = [ClassMap(r"D:\local\CNTK-2-0-rc1\cntk\Examples\Image\DataSets\Grocery\Class_map.txt")]
tree_map = TreeMap(cls_maps, use_background=True, only_valid_leafs=True, reduce_graph=True, use_multiply_with_parent=False)
params = par.get_parameters_for_dataset()

len(params.classes)

def get_vectors_for_label(label):
    index = np.argmax(label)
    return tree_map.get_train_softmax_vectors(to_Set=[(cls_maps[0], index)], scale_value=1)

def to_non_hierarchical_output(output):
    output = output.copy()
    if tree_map.use_background: # background is used on all branches which reduces the weight of each class. Therefore the effect of the background neuron needs to be removed so that the sum of the classes in the given branch is 1
        #unscore the background labels below!
        for r in range(1,len(tree_map.softmax_regions)):
            region=tree_map.softmax_regions[r]
            start=region[0]
            stop = region[1]
            b=output[:,:,start] # background value
            sum_c=np.add.reduce(output[:,:,start+1:stop], axis=2) # sum over classes
            mult=b/sum_c + 1
            mult.shape+=(1,)
            #import ipdb;ipdb.set_trace()
            output[:, :, start + 1:stop]*=mult

    nonh = np.zeros(output.shape[0:-1]+(cls_maps[0].entries,))

    for cls_i in range(cls_maps[0].entries):
        cls_string =cls_maps[0].getClass(cls_i)

        if not tree_map.use_multiply_with_parent:
            if not cls_string == "__background__":
                cls_node = tree_map.meta_map[cls_maps[0]][cls_string]
                multiply_index_list = [cls_node._index_in_network]
                cls_node = cls_node.prev[0]
                while not cls_node.is_root():
                    multiply_index_list.append(cls_node._index_in_network)
                    cls_node=cls_node.prev[0]

            for mb_i in range(output.shape[0]):
                for pred_i in range (output.shape[1]):
                    if  cls_string == "__background__":
                        if tree_map.use_background: nonh[mb_i, pred_i, cls_i] = output[mb_i, pred_i,0]
                        continue

                    value = 1
                    for m in range(len(multiply_index_list)):
                        value *= output[mb_i,pred_i,multiply_index_list[m]]
                    nonh[mb_i,pred_i,cls_i] = value
        else:
            assert False, "not implemented yet"





    return nonh



class Target_Creator(UserFunction): # usefoll for debugging...
    def __init__(self,
            arg,
            name='',):

        super(Target_Creator, self).__init__([arg], name=name)

    def infer_outputs(self):
        return [output_variable((params.cntk_nrRois, tree_map.get_nr_of_required_neurons()), self.inputs[0].dtype, self.inputs[0].dynamic_axes, needs_gradient=False)]

    def forward(self, argument, device=None, outputs_to_retain=None):
        mb_size= len(argument)
        output = np.zeros((mb_size, params.cntk_nrRois, tree_map.get_nr_of_required_neurons() ))
        for i in range(len(argument)):
            for j in range(len(argument[i])):
                target,_=get_vectors_for_label(argument[i][j])
                #assert target[0]==0
                output[i][j] = target
        #import ipdb;
        #ipdb.set_trace()

        return None, output

    def backward(self, state, root_gradients):
        return root_gradients

    def clone(self, cloned_inputs):
        return self.__class__(*cloned_inputs, name=self.name)