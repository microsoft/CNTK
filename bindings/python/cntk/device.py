# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from enum import Enum, unique
from . import cntk_py
from cntk.internal import typemap

@unique
class DeviceKind(Enum):
    '''
    Describes different device kinds like CPU or GPU.
    '''

    CPU = cntk_py.DeviceKind_CPU
    GPU = cntk_py.DeviceKind_GPU

    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other
        if isinstance(other, DeviceKind):
            return self.value == other.value
        return self == other

    def __ne__(self, other):
        return not (self == other)

class DeviceDescriptor(cntk_py.DeviceDescriptor):
    '''
    Describes a device by an unique id and its type. If the device corresponds
    to a GPU its type is 1, otherwise, it is 0
    '''

    def id(self):
        '''
        Returns id of device descriptor

        Returns:
            `int`: id
        '''
        return super(DeviceDescriptor, self).id()

    def type(self):
        '''
        Returns type of device descriptor. 1 if it is a GPU device or 0 if CPU.

        Returns:
            `int`: type
        '''
        return super(DeviceDescriptor, self).type()

    def __str__(self):
        if self.type() == DeviceKind.GPU:
            details = 'GPU %i' % self.id()
        else:
            details = 'CPU'
        return "Device %s" % details

    def is_locked(self):
        '''
        Returns `True` if another CNTK process already holds an exclusive lock on this device.
        '''
        return super(DeviceDescriptor, self).is_locked()

@typemap
def all_devices():
    '''
    Returns a device descriptor list with all the available devices

    Returns:
        :class:`~cntk.device.DeviceDescriptor` list: all device descriptors
    '''
    return cntk_py.DeviceDescriptor.all_devices()

@typemap
def cpu():
    '''
    Returns CPU device descriptor

    Returns:
        :class:`~cntk.device.DeviceDescriptor`: CPU device descriptor
    '''
    return cntk_py.DeviceDescriptor.cpu_device()

@typemap
def gpu(device_id):
    '''
    Returns GPU device descriptor

    Returns:
        :class:`~cntk.device.DeviceDescriptor`: GPU device descriptor
    '''
    return cntk_py.DeviceDescriptor.gpu_device(device_id)

@typemap
def use_default_device():
    '''
    Freezes the default device of the current CNTK process disallowing further changes
    through calls to :func:`try_set_default_device`. This default device will used for all 
    CNTK operations where a device needs to be specified and where none was explicitly 
    provided. If no device has been specified with a call to :func:`try_set_default_device`, 
    on the first invocation, this methods will auto-select one of the available (non-locked) 
    devices as the default.

    Returns:
        :class:`~cntk.device.DeviceDescriptor`: descriptor of the globally default device
    '''
    return cntk_py.DeviceDescriptor.use_default_device()

def set_default_device(new_default_device):
    '''
    See :func:`try_set_default_device`
    '''
    import warnings
    warnings.warn('This will be removed in future versions. Please use '
                  'DeviceDescriptor.try_set_default_device() instead.', DeprecationWarning)
    return try_set_default_device(new_default_device, False)

def try_set_default_device(new_default_device, acquire_device_lock=False):
    '''
    Tries to set the specified device as the globally default, optionally acquiring an exclusive 
    (cooperative) lock on the device (only a GPU device can be locked).

    Args:
        new_default_device (:class:`~cntk.device.DeviceDescriptor`): a descriptor of the device 
         to be used as a globally default.
        acquire_device_lock (bool, defaults to `False`): whether or not a lock should be acquired
         for the specified device.

    The default device can only be changed if it has not yet been frozen by being implicitly used 
    in any previous CNTK operation. 

    CNTK uses a cooperative synchronization for the device access, whereby only a single process 
    can acquire a device lock. However, if exclusivity is not required, the same device can still 
    be accessed without acquiring any locks (in which case, any existing lock corresponding to the
    device will be ignored).

    Returns: `False` if
        * the specified device appears in the list of excluded devices;
        * `acquire_device_lock` is `True` and another process already holds a lock on the device;
        * `acquire_device_lock` is `True` and `new_default_device` corresponds to a CPU device 
          (which cannot be locked).
    '''
    return cntk_py.DeviceDescriptor.try_set_default_device(new_default_device, acquire_device_lock)

def set_excluded_devices(excluded_devices):
    '''
    Allows to specify a list of excluded devices that cannot be used as globally default (neither 
    auto-selected nor explicitly specified by :func:`try_set_default_device`). For example, to avoid 
    auto-selecting the CPU, invoke ``set_excluded_devices([cpu()])``. However, after the default 
    device has been selected and frozen (by a call to :func:`use_default_device`), invoking this 
    methods has no effect, it becomes essentially a no-op.

    Args:
        excluded_devices (list of :class:`~cntk.device.DeviceDescriptor`): a list of device descriptors
         to exclude.
    '''
    cntk_py.DeviceDescriptor.set_excluded_devices(excluded_devices)

def get_gpu_properties(device):
    '''
    Retrieves and returns additional properties (total memory, number of CUDA cores, etc.) for 
    the specified GPU device. This method will raise an exception if a CPU device is specified 
    as an argument.

    Args:
        device (:class:`~cntk.device.DeviceDescriptor`): a GPU device descriptor.

     Returns:
        :class:`~cntk.cntk_py.GPUProperties`: GPU device properties
    '''
    return cntk_py.DeviceDescriptor.get_gpu_properties(device)
