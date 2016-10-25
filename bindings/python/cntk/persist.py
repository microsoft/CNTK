# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk import cntk_py
from .utils.swig_helper import typemap
from cntk.device import use_default_device

def save_model(root_op, filename, use_legacy_format=True):
    '''
    Save the network of ``root_op`` in ``filename``.

    Args:
        root_op (:class:`cntk.functions.Function`): op of the graph to save
        filename (`str`): filename to store the model in
        use_legacy_format (`str`): if 'True', model is stored using legacy format.
             Otherwise, it's stored using protobuf-based protocol serialization.
    '''
    root_op.save_model(filename, use_legacy_format)

@typemap
def load_model(data_type, filename, device=None):
    '''
    Load the network in ``filename``, that has been saved using
    `:func:save_model`.

    Args:
        data_type ('float' or 'double', or NumPy type): data type of the operation
        filename (`str`): filename to load the model from
        device (:class:`cntk.device.DeviceDescriptor`, default to default device): instance of DeviceDescriptor

    Returns:
        root node
    '''
    from cntk.utils import sanitize_dtype_cntk
    data_type = sanitize_dtype_cntk(data_type)
    if not device:
        device = use_default_device()
    return cntk_py.Function.load_model(data_type, filename, device)
