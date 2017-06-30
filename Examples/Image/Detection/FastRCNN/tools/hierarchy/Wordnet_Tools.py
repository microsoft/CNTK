# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import nltk
from nltk.corpus import wordnet as wn


def get_top_synsets(strings):
    out = {}
    for i in range(len(strings)):
        string = strings[i]
        syns = wn.synsets(string, wn.NOUN)

        out[string] = syns[0] if len(syns) > 0 else None

    return out