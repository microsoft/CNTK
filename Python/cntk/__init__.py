#
# CNTK proxy that translates Keras graphs into a CNTK configuration file.
#

import os
import numpy as np
from .context import *
from .graph import *
from .ops import *

_FLOATX = 'float32'

