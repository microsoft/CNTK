# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

__version__ = '2.0.beta15.0+'

import numpy as np

from .core import *
from . import ops
from . import cntk_py
from . import debugging
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
from .learner import *
# End of to remove

DATATYPE = np.float32

InferredDimension = cntk_py.InferredDimension
