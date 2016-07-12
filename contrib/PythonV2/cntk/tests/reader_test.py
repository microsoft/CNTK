# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import pytest

from ..reader import *
from ..graph import *
from ..context import *
from ..ops import constant, dynamic_axis

from cntk.tests.test_utils import *

# Keeping things short
C = constant
I = input

