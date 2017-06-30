# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import nltk
from nltk.corpus import wordnet as wn


class DAC_Node():
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
        DAC_Utils.print_Tree(self)

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

        cloned = DAC_Node()
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




class DAC_Utils():

    @staticmethod
    def get_new_root_node():
        node = DAC_Node()
        node.syn = wn.synsets("entity")[0]
        return node

    @staticmethod
    def add_synset_to_DAC(root, syn, cls_map, string):
        if root is None: return None
        if syn is None:
            # If the wordnet_graph had no match fpr the string, append to root!
            assert cls_map is not None and string is not None, "Nodes without wordnet informations cannot be added w/o content"
            node = DAC_Node()
            node.strings.append(string)
            node.cls_maps.append(cls_map)
            node.prev.append(root)
            root.next.append(node)
            return node

        if root.syn == syn: return root

        # Check whether parents are in the graph and add them if necessary!
        hypernyms = syn.hypernyms()
        if not hypernyms:
            # an empty list could have been returned!
            #
            # so try to get hypernyms via hypernympaths!
            hyp_paths = syn.hypernym_paths()
            if len(hyp_paths[0])>1 :
                for i in range(len(hyp_paths)):
                    hypa_len = len(hyp_paths[i])
                    if hypa_len > 1:
                        hypernyms.append(hyp_paths[i][hypa_len-2])
            if not hypernyms: # if even that fails, set it under root!
                hypernyms.append(root.syn)
        parent_nodes = []
        for i in range(len(hypernyms)):
            parent_nodes.append(DAC_Utils.add_synset_to_DAC(root, hypernyms[i], None, None))

        new_node = None
        for i in range(len(parent_nodes)):
            # search for existing node of this item in the parents
            children = parent_nodes[i].next
            for j in range(len(children)):
                if hasattr(children[j], 'syn') and children[j].syn == syn:
                    new_node = children[j]
                    break
            if new_node is not None:
                break

        # if node is not existing yet
        if new_node is None:
            new_node = DAC_Node()
            new_node.syn = syn
            new_node.prev = parent_nodes
            if string is not None: new_node.strings.append(string)
            if cls_map is not None: new_node.cls_maps.append(cls_map)

            # ensure node is in every parent
            for i in range(len(parent_nodes)):
                parent_nodes[i].next.append(new_node)

        return new_node

    @staticmethod
    def print_Tree(tree, indent=""):
        content = str(tree.syn) if hasattr(tree, "syn") else "Raw_Node: " + tree.strings[0]
        print(indent + content)
        for i in range(len(tree.next)):
            DAC_Utils.print_Tree(tree.next[i], indent + "|  ")