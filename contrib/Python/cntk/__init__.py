# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

__version__ = '1.4'

from .context import *
from .graph import *
from . import ops
from .sgd import *
from .reader import UCIFastReader, CNTKTextFormatReader
from .ops import *
from .utils.eval import eval
