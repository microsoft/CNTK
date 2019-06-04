# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
os.environ["PATH"] += os.pathsep + os.path.join(os.path.dirname(__file__), 'libs')
cntk_library_path=os.pathsep + os.path.join(os.path.dirname(__file__), 'libs')
if os.environ.get('LD_LIBRARY_PATH'):
    os.environ['LD_LIBRARY_PATH'] += cntk_library_path
else:
    os.environ['LD_LIBRARY_PATH'] = cntk_library_path

# Read version information
version_file = open(os.path.join(os.path.dirname(__file__), 'VERSION'), 'r')
__version__ = version_file.read()
version_file.close()
del version_file

# Test minimum requirements before running
from . import cntk_py_init
cntk_py_init.cntk_check_distro_info()
cntk_py_init.cntk_check_libs()

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
from . import misc
from . import random

from .sample_installer import install_samples

DATATYPE = np.float32
InferredDimension = cntk_py.InferredDimension
FreeDimension = cntk_py.FreeDimension

from .internal.utils import _to_cntk_dict_value
import _cntk_py
cntk_py.Dictionary.__setitem__ = lambda self, key, value: _cntk_py.Dictionary___setitem__(self, key, _to_cntk_dict_value(value))