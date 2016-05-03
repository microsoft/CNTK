# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Utils for operations unit tests
"""

import numpy as np
import pytest

from cntk.tests.test_utils import *

from ...context import get_new_context
from ...reader import *
from ..variables_and_parameters import *


# Keeping things short
C = constant
I = input_reader
