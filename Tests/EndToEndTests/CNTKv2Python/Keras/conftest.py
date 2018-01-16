# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import zipfile
import shutil
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve
import cntk
from cntk.ops.tests.ops_test_utils import cntk_device

abs_path = os.path.dirname(os.path.abspath(__file__))

keras_version = os.environ['KERAS_TEST_VERSION']

keras_base_name = 'keras-%s' % (keras_version)
keras_zip_name = keras_base_name + '.zip'
keras_zip_path = os.path.join(abs_path, keras_zip_name)
keras_path = os.path.join(abs_path, keras_base_name)

if not os.path.exists(keras_path):
  if not os.path.exists(keras_zip_path):
    if 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY' in os.environ:
      shutil.copy(os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'], 'Keras', keras_zip_name), abs_path)
    else:
      keras_zip_url = 'https://github.com/fchollet/keras/archive/%s.zip' % (keras_version)
      urlretrieve(keras_zip_url, keras_zip_path)
  with zipfile.ZipFile(keras_zip_path) as keras_zip:
    keras_zip.extractall(abs_path)
  # We'll use our own pytest.ini, move original out of the way
  os.rename(os.path.join(keras_path, 'pytest.ini'), os.path.join(keras_path, 'pytest.ini.bak'))

cntk_test_device_id = -1 if os.environ.get('TEST_DEVICE', 'cpu') == 'cpu' else 0
cntk.device.try_set_default_device(cntk_device(cntk_test_device_id))

# Files that we can't even import (we don't install all of the dependencies, e.g., tensorflow)
collect_ignore = [os.path.join(*[keras_base_name, 'tests', 'keras', 'backend', 'backend_test.py'])]
