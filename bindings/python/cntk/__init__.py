# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

__version__ = '2.0'

import os

from . import graph
from . import ops
from .cntk_py import *

import numpy as np

DATATYPE = np.float32
