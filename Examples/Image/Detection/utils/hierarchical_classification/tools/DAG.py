# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
#
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

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
        node.strings.append("entity")
        return node


    @staticmethod
    def print_Tree(tree, indent=""):
        content = "Node: " + (tree.strings[0] if tree.strings else "unnamed")
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
                    done_map[lnode]=value
                    value +=1
                    todo.extend(lnode.next)
                    if multiple_roots: todo.extend(node.prev)
            return value, done_map

        def _remove_serialization_indicies(node):
            if node and hasattr(node, "__serialize_id"):
                for child in node.next:
                    _remove_serialization_indicies(child)
                if multiple_roots:
                    for parent in node.prev:
                        _remove_serialization_indicies(parent)

        def _get_node_repr(node, id_map):
            # use JSON style where each node is referenced by an id
            node_repr={}
            node_repr["id"] =id_map[node]
            node_repr["strings"]=node.strings
            node_repr["cls_maps"]=[]
            for cls_map in node.cls_maps:
                node_repr["cls_maps"].append(cls_map.file)
            node_repr["childrens"]=[]
            for child in node.next:
                node_repr["childrens"].append(id_map[child])
            return node_repr

        node_count, node_map=_apply_serialization_indices(node, 0)
        dac_repr_list = [None]*node_count
        todo=[node]
        while todo:
           lnode = todo.pop()
           ind =node_map[lnode]
           if dac_repr_list[ind] is None:
               dac_repr_list[ind]=_get_node_repr(lnode, node_map)
               todo.extend(lnode.next)
               if multiple_roots:
                   todo.extend(lnode.prev)

        return dumps(dac_repr_list)


    @staticmethod
    def deserialize(string):

        def _create_DAG_Node_from_repr(repr, cls_maps):
            node = DAG_Node()
            node.strings.extend(repr["strings"])

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
