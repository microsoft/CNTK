# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

__version__ = '2.0'

import os
# required for the _cntk_py.pyd/so file
os.environ['PATH'] += ';'+os.path.abspath(__file__)

from .graph import *
from . import ops
from .ops import *


import numpy as np

DATATYPE = np.float32
