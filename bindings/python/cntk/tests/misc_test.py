# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import pytest
import cntk
from cntk.device import *
import sys
from multiprocessing import Process, Queue


cntk_py.always_allow_setting_default_device()

def is_locked_cross_process(queue, device_id):
    device = cpu() if device_id < 0 else gpu(device_id)
    queue.put(device.is_locked())

def is_locked(device):
    q = Queue()
    device_id = -1 if (device.type() == DeviceKind.CPU) else device.id();
    p = Process(target=is_locked_cross_process, args=(q,device_id))
    p.start()
    p.join()
    assert p.exitcode == 0
    return q.get()

def test_callstack1():
    with pytest.raises(ValueError) as excinfo:
        cntk.device.gpu(99999)
    assert '[CALL STACK]' in str(excinfo.value)

def test_callstack2():
    with pytest.raises(ValueError) as excinfo:
        cntk.io.MinibatchSource(cntk.io.CTFDeserializer("", streams={}))
    assert '[CALL STACK]' in str(excinfo.value)

def test_Value_raises():
    from cntk import NDArrayView, Value
    with pytest.raises(ValueError):
        nd = NDArrayView.from_dense(np.asarray([[[4,5]]], dtype=np.float32))
        val = Value(nd)

def test_cpu_and_gpu_devices():
    device = cpu()
    assert device.type() == DeviceKind.CPU
    assert device.id() == 0
    for i in range(len(all_devices()) - 1):
        device = gpu(i)
        assert device.type() == DeviceKind.GPU
        assert device.id() == i

def test_all_devices():
    assert len(all_devices()) > 0
    assert cpu() in all_devices()
    if (len(all_devices()) > 1):
        assert gpu(0) in all_devices()

def test_gpu_properties():
    for device in all_devices():
        if (device.type() != DeviceKind.GPU):
            continue
        props =  get_gpu_properties(device)
        assert props.device_id == device.id()
        assert props.cuda_cores > 0
        assert props.total_memory > 0
        assert props.version_major > 0

def _use_default_device(queue):
    # use_default_device needs to be tested in isolation
    # in a freshly created process environment.
    device = use_default_device()
    if (device.type() != DeviceKind.GPU):
        queue.put(not is_locked(device))
    else:
        queue.put(is_locked(device))

def test_use_default_device():
    # this will release any previous held device locks
    try_set_default_device(cpu(), False)
    q = Queue()
    p = Process(target=_use_default_device, args=(q,))
    p.start()
    p.join()
    assert p.exitcode == 0
    assert q.get()

def test_set_cpu_as_default_device():
    device = cpu()
    assert not is_locked(device)
    assert not try_set_default_device(device, True)
    assert not is_locked(device)
    assert try_set_default_device(device)
    assert try_set_default_device(device, False)
    assert not is_locked(device)
    assert device == use_default_device()

def test_set_gpu_as_default_device():
  if len(all_devices()) == 1: 
      return;
  # this will release any previous held device locks
  try_set_default_device(cpu(), False)
  for i in range(len(all_devices()) - 1):
    device = gpu(i)
    assert try_set_default_device(device, False)
    assert not is_locked(device)
    assert device == use_default_device()
    if not device.is_locked():
        assert not is_locked(device)
        assert try_set_default_device(device, True)
        assert device == use_default_device()
        assert is_locked(device)

def test_set_excluded_devices():
  if len(all_devices()) == 1: 
      return;
  assert try_set_default_device(cpu(), False)
  assert try_set_default_device(gpu(0), False)
  set_excluded_devices([cpu()])
  assert not try_set_default_device(cpu(), False)
  set_excluded_devices([])
  assert try_set_default_device(cpu(), False)

def test_setting_trace_level():
  from cntk.logging import TraceLevel, set_trace_level, get_trace_level

  value = get_trace_level();
  assert value == TraceLevel.Warning
  
  for level in [TraceLevel.Info, TraceLevel.Error, TraceLevel.Warning]:
    set_trace_level(level)
    value = get_trace_level();
    assert value == level
    set_trace_level(level.value)
    value = get_trace_level();
    assert value == level