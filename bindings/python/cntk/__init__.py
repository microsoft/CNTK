# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

__version__ = '2.0.beta12.0+'

import os
import numpy as np

from .core import *
from . import ops
from . import cntk_py
from .train import *
from .learners import *
from .losses import *
from .metrics import *
from .initializer import *
from .utils import *
from .ops import *
from .device import *
from .layers import *
from .sample_installer import install_samples

# To __remove__
from .io import *

def one_hot(batch, num_classes, dtype=None, device=None):
    return Value.one_hot(batch, num_classes, dtype, device)
# End of to remove

DATATYPE = np.float32
InferredDimension = cntk_py.InferredDimension
