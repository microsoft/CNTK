# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================


class ClassMap():
    """Awaits format [<Identifier> <number> ]"""

    def __init__(self, cls_map_file, name=None):
        """
        Initialises this ClassMap object with the content of the file.
        :param cls_map_file: str containing an existing Path to a class map file
        :param name: optional name for this ClassMap, may be used for tagging
        """
        self.file = cls_map_file
        self.name = name
        strings = open(cls_map_file).read().split()
        self.cls2nr_map = {}
        self.entries = int(len(strings) / 2)
        self.cls_map = [None] * self.entries
        for i in range(self.entries):
            index = int(strings[2 * i + 1])
            label = strings[2 * i]
            self.cls_map[index] = label
            self.cls2nr_map[label] = index

    def getClass(self, i):
        """
        Getter method to get the class assigned to the label i
        :param i: label as int
        :return: str with classname
        """
        return self.cls_map[i]

    def getIndex(self, string):
        """
        Getter method to receive the label/index of to a classname
        :param string: classname as str
        :return: int representig the label
        """
        return self.cls2nr_map[string]

    def getEntries(self):
        """
        Getter Method to get the number of classes in this classmap
        :return: int
        """
        return self.entries
