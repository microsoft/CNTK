# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from . import cntk_py

class DeviceDescriptor(cntk_py.DeviceDescriptor):
    '''
    Describes a device by an unique id and its type. If the device corresponds to a GPU its type is 1,
    otherwise, it is 0
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

def all_devices():
    '''
    Returns a device descriptor list with all the available devices

    Returns:
        :class:`cntk.device.DeviceDescriptor` list: all device descriptors
    '''
    return cntk_py.DeviceDescriptor.all_devices()

def best():
    '''
    Returns a device descriptor with the best configuration.

    Returns:
        :class:`cntk.device.DeviceDescriptor`: Best device descriptor
    '''
    return cntk_py.DeviceDescriptor.best_device()

def cpu():
    '''
    Returns CPU device descriptor

    Returns:
        :class:`cntk.device.DeviceDescriptor`: CPU device descriptor
    '''
    return cntk_py.DeviceDescriptor.cpu_device()

def default():
    '''
    Returns default device

    Returns:
        :class:`cntk.device.DeviceDescriptor`: Default device descriptor
    '''
    return cntk_py.DeviceDescriptor.default_device()

def gpu(device_id):
    '''
    Returns GPU device

    Returns:
        :class:`cntk.device.DeviceDescriptor`: GPU device descriptor
    '''
    return cntk_py.DeviceDescriptor.gpu_device(device_id)

def use_default_device():
    '''
    Use default device

    Returns:
        `int`: Id of default device
    '''
    return cntk_py.DeviceDescriptor.use_default_device()

def set_default_device(new_default_device):
    '''
    Set new device descriptor as default

    Args:
        new_default_device (:class:`cntk.device.DeviceDescriptor`): new device descriptor

    Returns:
        :class:`cntk.device.DeviceDescriptor`: id
    '''
    return cntk_py.DeviceDescriptor.set_default_device(new_default_device)
