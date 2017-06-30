# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================


class ClassMap():
    """Awaits format {<Identifier> <number> }"""

    def __init__(self, cls_map_file, name=None):
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
        return self.cls_map[i]

    def getIndex(self, string):
        return self.cls2nr_map[string]

    def getEntries(self):
        return self.entries