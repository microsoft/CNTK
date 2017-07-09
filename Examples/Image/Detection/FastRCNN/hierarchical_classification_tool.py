
from tools.hierarchy.Tree_Creator import TreeMap
from tools.ClassMap import ClassMap
import sys, path, os
import PARAMETERS as par
import numpy as np
from cntk.ops.functions import UserFunction
from cntk import *

my_path = os.path.dirname(os.path.realpath(__file__))
cls_maps = [ClassMap(my_path + r"/../../DataSets/Grocery/Class_map.txt")]
tree_map = TreeMap.tree_map_from_cls_maps(cls_maps, use_background=True, only_valid_leafs=True, reduce_graph=True, use_multiply_with_parent=False)
params = par.get_parameters_for_dataset()


flat_tree_grocery_str = '[' \
    '   {"id": 0, "childrens": [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], "syn": "Synset(\'physical_entity.n.01\')", "strings": [], "cls_maps": []}, ' \
    '   {"id": 1, "childrens": [], "syn": "Synset(\'water.n.01\')", "strings": ["water"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 2, "childrens": [], "syn": "Synset(\'champagne.n.01\')", "strings": ["champagne"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 3, "childrens": [], "syn": "Synset(\'juice.n.01\')", "strings": ["juice"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 4, "childrens": [], "syn": "Synset(\'catsup.n.01\')", "strings": ["ketchup"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 5, "childrens": [], "syn": "Synset(\'gherkin.n.01\')", "strings": ["gherkin"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 6, "childrens": [], "syn": "Synset(\'egg.n.02\')", "strings": ["eggs"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 7, "childrens": [], "syn": "Synset(\'milk.n.01\')", "strings": ["milk"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 8, "childrens": [], "syn": "Synset(\'yogurt.n.01\')", "strings": ["yoghurt"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 9, "childrens": [], "syn": "Synset(\'butter.n.01\')", "strings": ["butter"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 10, "childrens": [], "syn": "Synset(\'tomato.n.01\')", "strings": ["tomato"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 11, "childrens": [], "syn": "Synset(\'tabasco.n.01\')", "strings": ["tabasco"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 12, "childrens": [], "syn": "Synset(\'mustard.n.01\')", "strings": ["mustard"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 13, "childrens": [], "syn": "Synset(\'pepper.n.01\')", "strings": ["pepper"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 14, "childrens": [], "syn": "Synset(\'onion.n.01\')", "strings": ["onion"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 15, "childrens": [], "syn": "Synset(\'orange.n.01\')", "strings": ["orange"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 16, "childrens": [], "syn": "Synset(\'avocado.n.01\')", "strings": ["avocado"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}' \
    ']'

if False:
    tree_map = TreeMap.tree_map_from_tree_str(flat_tree_grocery_str, use_background=True, use_multiply_with_parent=False)
    cls_maps = list(tree_map.meta_map.keys())


tree_map.root_node.print()
#len(params.classes)

def get_vectors_for_label(label):
    index = np.argmax(label)
    return tree_map.get_train_softmax_vectors(to_Set=[(cls_maps[0], index)], scale_value=1)





def top_down_eval(vector):
    out_vec = np.zeros(vector.shape, dtype=np.float32)

    start = 0
    multiplier = 1

    # switch to tree descending to get log(N) instead of n
    for region in tree_map.softmax_regions:
        if start < region[0]: continue

        node_index = np.argmax(vector[region[0]:region[1]], axis=0) + region[0]
        #if np.add.reduce(out_vec)==0 and np.argmax(vector[0:3])!=0: import ipdb;ipdb.set_trace()
        node = tree_map.as_in_network[node_index]
        if node is None: # background --> stop here!
            if node_index == 0:# first bg --> set global bg
                out_vec[0] = vector[0]
                assert (out_vec[0]!=0 or vector[0]==0)
            break

        if tree_map.use_background:
            bg_value = vector[region[0]]
            bg_removal_factor = 1 / (1 - bg_value)
            multiplier *= bg_removal_factor

        multiplier *= vector[node_index]
        out_vec[node_index] = multiplier

        if not node.next: # leaf
            #import ipdb;ipdb.set_trace()
            assert np.add.reduce(out_vec) > 0
            break


        # else
        start = node.next[0]._index_in_network

    if np.add.reduce(out_vec)==0: import ipdb;ipdb.set_trace()
    return out_vec

    # labels = []
    # start = -1
    # # switch to tree descending to get log(N) instead of n
    # for region in tree_map.softmax_regions:
    #     if start > region[0]: continue
    #
    #     node_index = np.argmax(vector[region[0]:region[1]], axis=0) + region[0]
    #
    #     node = tree_map.as_in_network[node_index]
    #     if node is None:
    #         if node_index == 0:
    #             labels.append(node_index)
    #         break
    #     if not node.next:
    #         break
    #
    #     labels.append(node_index)
    #
    #     start = node.next[0]


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

if __name__ == "__main__":
    from tools.hierarchy.Syn_DAC import DAC_Utils
    tree_str = DAC_Utils.serialize(tree_map.root_node)
    tree_cpy = DAC_Utils.deserialize(tree_str)

    

    # auto generated tree_map
    '[' \
    '   {"id": 0, "childrens": [16, 1], "syn": "Synset(\'physical_entity.n.01\')", "strings": [], "cls_maps": []}, {"id": 1, "childrens": [12, 3, 2], "syn": "Synset(\'matter.n.03\')", "strings": [], "cls_maps": []}, ' \
    '   {"id": 2, "childrens": [], "syn": "Synset(\'water.n.01\')", "strings": ["water"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 3, "childrens": [5, 4], "syn": "Synset(\'food.n.01\')", "strings": [], "cls_maps": []}, ' \
    '   {"id": 4, "childrens": [], "syn": "Synset(\'champagne.n.01\')", "strings": ["champagne"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 5, "childrens": [11, 10, 7, 6], "syn": "Synset(\'foodstuff.n.02\')", "strings": [], "cls_maps": []}, ' \
    '   {"id": 6, "childrens": [], "syn": "Synset(\'juice.n.01\')", "strings": ["juice"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 7, "childrens": [9, 8], "syn": "Synset(\'condiment.n.01\')", "strings": [], "cls_maps": []}, ' \
    '   {"id": 8, "childrens": [], "syn": "Synset(\'catsup.n.01\')", "strings": ["ketchup"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 9, "childrens": [], "syn": "Synset(\'gherkin.n.01\')", "strings": ["gherkin"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 10, "childrens": [], "syn": "Synset(\'egg.n.02\')", "strings": ["eggs"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 11, "childrens": [], "syn": "Synset(\'milk.n.01\')", "strings": ["milk"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 12, "childrens": [15, 14, 13], "syn": "Synset(\'food.n.02\')", "strings": [], "cls_maps": []}, ' \
    '   {"id": 13, "childrens": [], "syn": "Synset(\'yogurt.n.01\')", "strings": ["yoghurt"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 14, "childrens": [], "syn": "Synset(\'butter.n.01\')", "strings": ["butter"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 15, "childrens": [], "syn": "Synset(\'tomato.n.01\')", "strings": ["tomato"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 16, "childrens": [18, 17], "syn": "Synset(\'object.n.01\')", "strings": [], "cls_maps": []}, ' \
    '   {"id": 17, "childrens": [], "syn": "Synset(\'tabasco.n.01\')", "strings": ["tabasco"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 18, "childrens": [22, 19], "syn": "Synset(\'whole.n.02\')", "strings": [], "cls_maps": []}, ' \
    '   {"id": 19, "childrens": [21, 20], "syn": "Synset(\'vascular_plant.n.01\')", "strings": [], "cls_maps": []}, ' \
    '   {"id": 20, "childrens": [], "syn": "Synset(\'mustard.n.01\')", "strings": ["mustard"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 21, "childrens": [], "syn": "Synset(\'pepper.n.01\')", "strings": ["pepper"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 22, "childrens": [24, 23], "syn": "Synset(\'plant_organ.n.01\')", "strings": [], "cls_maps": []}, ' \
    '   {"id": 23, "childrens": [], "syn": "Synset(\'onion.n.01\')", "strings": ["onion"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 24, "childrens": [26, 25], "syn": "Synset(\'edible_fruit.n.01\')", "strings": [], "cls_maps": []}, ' \
    '   {"id": 25, "childrens": [], "syn": "Synset(\'orange.n.01\')", "strings": ["orange"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}, ' \
    '   {"id": 26, "childrens": [], "syn": "Synset(\'avocado.n.01\')", "strings": ["avocado"], "cls_maps": ["' + my_path + '/../../DataSets/Grocery/Class_map.txt"]}' \
    ']'

    

    import ipdb;

    ipdb.set_trace()
