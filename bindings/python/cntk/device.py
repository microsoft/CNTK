# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from . import cntk_py

class DeviceDescriptor(cntk_py.DeviceDescriptor):
    def all_devices():
        '''
        Returns a device descriptor list with all the available devices

        Returns:
            :class:`cntk.device.DeviceDescriptor` list: all device descriptors
        '''
        return cntk_py.DeviceDescriptor.all_devices()

    def best_device():
        '''
        Returns a device descriptor with the best configuration.

        Returns:
            :class:`cntk.device.DeviceDescriptor`: Best device descriptor
        '''
        return cntk_py.DeviceDescriptor.best_device()

    def cpu_device():
        '''
        Returns CPU device descriptor

        Returns:
            :class:`cntk.device.DeviceDescriptor`: CPU device descriptor
        '''
        return cntk_py.DeviceDescriptor.cpu_device()

    def default_device():
        '''
        Returns default device

        Returns:
            :class:`cntk.device.DeviceDescriptor`: Default device descriptor
        '''
        return cntk_py.DeviceDescriptor.default_device()

    def gpu_device(deviceId):
        '''
        Returns GPU device

        Returns:
            :class:`cntk.device.DeviceDescriptor`: GPU device descriptor
        '''
        return cntk_py.DeviceDescriptor.gpu_device(deviceId)

    def id(self):
        '''
        Returns id of device descriptor

        Returns:
            `int`: id
        '''
        return super(DeviceDescriptor, self).id()

    def set_default_device(newDefaultDevice):
        '''
        Set new device descriptor as default

        Args:
            newDefaultDevice (:class:`cntk.device.DeviceDescriptor`): new device descriptor

        Returns:
            :class:`cntk.device.DeviceDescriptor`: id
        '''
        return cntk_py.DeviceDescriptor.set_default_device(newDefaultDevice)

    def type(self):
        '''
        Returns type of device descriptor. 1 if it is a GPU device or 0 if CPU.

        Returns:
            `int`: type
        '''
        return super(DeviceDescriptor, self).type()

    def use_default_device():
        '''
        Use default device

        Returns:
            `int`: Id of default device
        '''
        return cntk_py.DeviceDescriptor.use_default_device()

def DeviceDescriptor_eq(first, second):
    return cntk_py.DeviceDescriptor_eq(first, second)

DeviceDescriptor.__eq__ = lambda a,b: DeviceDescriptor_eq(a,b)