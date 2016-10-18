# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

__version__ = '2.0'

import os
import numpy as np

from . import ops
from . import cntk_py

from .trainer import *
from .learner import *
from .initializer import *
from .utils import *
from .ops import *
from .io import *
from .persist import load_model, save_model
from .device import *

# TODO wrap
from .cntk_py import momentums_per_sample

DATATYPE = np.float32
