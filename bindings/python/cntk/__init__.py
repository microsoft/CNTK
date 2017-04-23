# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

__version__ = '2.0rc1+'

import numpy as np

from . import cntk_py

#
# Bubble the below namespaces to cntk root namespace.
#
from .core import *
from .variables import Parameter, Constant
from .ops import *
from .device import *
from .train import *
from .eval import *
from .learners import *
from .losses import *
from .metrics import *
from .initializer import *
from .default_options import *

from . import debugging
from . import logging
from . import io
from . import layers

from .sample_installer import install_samples

DATATYPE = np.float32
InferredDimension = cntk_py.InferredDimension
