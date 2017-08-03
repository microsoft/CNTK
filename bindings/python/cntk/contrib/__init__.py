# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Extra utilities for CNTK, e.g. utilities that bridge to other deep learning toolkits.
"""


import numpy as np

from . import crosstalk
from . import crosstalkcaffe

#note that crosstalk_* is not imported here to reduce load time
