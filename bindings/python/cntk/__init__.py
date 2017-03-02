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

from .trainer import *
from .learner import *
from .initializer import *
from .utils import *
from .ops import *
from .io import *
from .debug import save_as_legacy_model
from .device import *
from .layers import *
from .distributed import *
from .training_session import *
from .sample_installer import install_samples

DATATYPE = np.float32
InferredDimension = cntk_py.InferredDimension
