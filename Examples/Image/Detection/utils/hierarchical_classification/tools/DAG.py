# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

#import nltk
#from nltk.corpus import wordnet as wn
from json import dumps, loads
from tools.ClassMap import ClassMap


class DAG_Node():
    def __init__(self):
        # list of previous nodes
        self.prev = []
        # list of following nodes
        self.next = []
        # list of cls_maps responsible
        self.cls_maps = []
        # string entered
        self.strings = []
        # synset to contain
        syn = None

    def print(self):
        DAG_Utils.print_Tree(self)

    def get_root(self):
        node = self
        while len(node.prev)>0:
            node = node.prev[0]
        return node

    def is_root(self):
        return len(self.prev)==0

    def __remove_clonedinstance__(self):
        if hasattr(self, "cloned_instance"): self.cloned_instance=None
        for i in range(len(self.next)):
            self.next[i].__remove_clonedinstance__()

    def __clone_helper__(self):
        if hasattr(self, "cloned_instance"):
            return self.cloned_instance

        cloned = DAG_Node()
        self.cloned_instance = cloned

        cloned.strings = self.strings.copy()
        cloned.cls_maps = self.cls_maps.copy()

        for i in range(len(self.next)):
            new_child = self.next[i].__clone_helper__()
            new_child.prev.append(cloned)
            cloned.next.append(new_child)

    def clone(self):
        """
        Builds a copy of the DAC starting with this node as root
        :return:
        """
        cloned = self.__clone_helper__()
        self.__remove_clonedinstance__()
        return cloned




class DAG_Utils():

    @staticmethod
    def get_new_root_node():
        node = DAG_Node()
        #node.syn = wn.synsets("entity")[0]
        node.strings.append("entity")
        return node

    # @staticmethod
    # def add_synset_to_DAG(root, syn, cls_map, string):
    #     if root is None: return None
    #     if syn is None:
    #         # If the wordnet_graph had no match fpr the string, append to root!
    #         assert cls_map is not None and string is not None, "Nodes without wordnet informations cannot be added w/o content"
    #         node = DAG_Node()
    #         node.strings.append(string)
    #         node.cls_maps.append(cls_map)
    #         node.prev.append(root)
    #         root.next.append(node)
    #         return node
    #
    #     if root.syn == syn: return root
    #
    #     # Check whether parents are in the graph and add them if necessary!
    #     hypernyms = syn.hypernyms()
    #     if not hypernyms:
    #         # an empty list could have been returned!
    #         #
    #         # so try to get hypernyms via hypernympaths!
    #         hyp_paths = syn.hypernym_paths()
    #         if len(hyp_paths[0])>1 :
    #             for i in range(len(hyp_paths)):
    #                 hypa_len = len(hyp_paths[i])
    #                 if hypa_len > 1:
    #                     hypernyms.append(hyp_paths[i][hypa_len-2])
    #         if not hypernyms: # if even that fails, set it under root!
    #             hypernyms.append(root.syn)
    #     parent_nodes = []
    #     for i in range(len(hypernyms)):
    #         parent_nodes.append(DAG_Utils.add_synset_to_DAG(root, hypernyms[i], None, None))
    #
    #     new_node = None
    #     for i in range(len(parent_nodes)):
    #         # search for existing node of this item in the parents
    #         children = parent_nodes[i].next
    #         for j in range(len(children)):
    #             if hasattr(children[j], 'syn') and children[j].syn == syn:
    #                 new_node = children[j]
    #                 break
    #         if new_node is not None:
    #             break
    #
    #     # if node is not existing yet
    #     if new_node is None:
    #         new_node = DAG_Node()
    #         new_node.syn = syn
    #         new_node.prev = parent_nodes
    #         if string is not None: new_node.strings.append(string)
    #         if cls_map is not None: new_node.cls_maps.append(cls_map)
    #
    #         # ensure node is in every parent
    #         for i in range(len(parent_nodes)):
    #             parent_nodes[i].next.append(new_node)
    #
    #     return new_node

    @staticmethod
    def print_Tree(tree, indent=""):
        #TODO remove syn
        content = str(tree.syn) if hasattr(tree, "syn") and tree.syn is not None else "Node: " + (tree.strings[0] if tree.strings else "unnamed")
        print(indent + content)
        for i in range(len(tree.next)):
            DAG_Utils.print_Tree(tree.next[i], indent + "|  ")


    @staticmethod
    def serialize(node, multiple_roots=True):
        def _apply_serialization_indices(node, value):
            done_map={}
            todo = [node]
            while todo:
                lnode = todo.pop()
                if not lnode in done_map:
                #if not hasattr(lnode, "__serialize_id"):
                #    lnode.__serialize_id = value
                    done_map[lnode]=value
                    value +=1
                    todo.extend(lnode.next)
                    if multiple_roots: todo.extend(node.prev)
            return value, done_map

        def _remove_serialization_indicies(node):
            if node and hasattr(node, "__serialize_id"):
                #node.__serialize_id = None
                for child in node.next:
                    _remove_serialization_indicies(child)
                if multiple_roots:
                    for parent in node.prev:
                        _remove_serialization_indicies(parent)

        def _get_node_repr(node, id_map):
            # use JSON style where each node is referenced by an id
            node_repr={}
            #node["id"]=node.__serialize_id
            node_repr["id"] =id_map[node]
            node_repr["strings"]=node.strings
            #node_repr["syn"]=str(node.syn)
            node_repr["cls_maps"]=[]
            for cls_map in node.cls_maps:
                node_repr["cls_maps"].append(cls_map.file)
            node_repr["childrens"]=[]
            for child in node.next:
                #node_repr["childrens"].append(child.__serialize_id)
                node_repr["childrens"].append(id_map[child])
            return node_repr

        node_count, node_map=_apply_serialization_indices(node, 0)
        dac_repr_list = [None]*node_count
        todo=[node]
        while todo:
           lnode = todo.pop()
           #if dac_repr_list[lnode.__serialize_id] is None:
           #    dac_repr_list[lnode.__serialize_id]=_get_node_repr(lnode)
           ind =node_map[lnode]
           if dac_repr_list[ind] is None:
               dac_repr_list[ind]=_get_node_repr(lnode, node_map)
               todo.extend(lnode.next)
               if multiple_roots:
                   todo.extend(lnode.prev)
        #_remove_serialization_indicies(node)

        return dumps(dac_repr_list)


    @staticmethod
    def deserialize(string):
        #def _find_syn_by_str(syn_str):
        #    return None

        def _create_DAG_Node_from_repr(repr, cls_maps):
            node = DAG_Node()
            node.strings.extend(repr["strings"])
            #node.syn = _find_syn_by_str(repr["syn"])
            #if node.syn is None and not node.strings and repr["syn"] is not None:
            #    node.strings.append(repr["syn"])

            # recreate cls_maps too
            for cls_map_str in repr["cls_maps"]:
                found = None
                for cls_map in cls_maps:
                    if cls_map.file == cls_map_str:
                        found=cls_map
                        break

                if found is None:
                    cls_map = ClassMap(cls_map_str)
                    cls_maps.append(cls_map)

                node.cls_maps.append(cls_map)
            return node


        dac_repr_list = loads(string)
        nr_of_nodes = len(dac_repr_list)
        cls_maps=[]
        dac_list = [None]*nr_of_nodes
        # create nodes
        for i in range(nr_of_nodes):
            dac_list[i]=_create_DAG_Node_from_repr(dac_repr_list[i], cls_maps)

        # create double links
        for i in range(nr_of_nodes):
            for j in dac_repr_list[i]["childrens"]:
                dac_list[i].next.append(dac_list[j])
                dac_list[j].prev.append(dac_list[i])

        root_node = dac_list[0].get_root() if dac_list else None

        return root_node, cls_maps



