# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os, sys
import pytest

run_if_win35_linux35_linux36 = pytest.mark.skipif(not ((sys.platform == 'win32' and sys.version_info[:2] == (3,5)) or
                                                       (sys.platform != 'win32' and sys.version_info[:2] == (3,5)) or
                                                       (sys.platform != 'win32' and sys.version_info[:2] == (3,6))),
                                                  reason="This test runs currently only in windows-py35, linux-py35 and linux-py36 due to precompiled cython modules")
