# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from .. import cntk_py


def start_profiler(dir='profiler', sync_gpu=True, reserve_mem=cntk_py.default_profiler_buffer_size):
    '''
    Start profiler to prepare performance statistics gathering. Note that
    the profiler is not enabled after start
    (`example
    <https://github.com/Microsoft/CNTK/wiki/Performance-Profiler#for-python>`_).

    Args:
        dir: directory for profiler output
        sync_gpu: whether profiler syncs CPU with GPU when timing
        reserve_mem: size in byte for profiler memory reserved
    '''
    cntk_py.start_profiler(dir, sync_gpu, reserve_mem)


def stop_profiler():
    '''
    Stop profiler from gathering performance statistics and flush them to file
    '''
    cntk_py.stop_profiler()


def enable_profiler():
    '''
    Enable profiler to gather data. Note that in training_session, profiler would be enabled automatically after the first check point
    '''
    cntk_py.enable_profiler()


def disable_profiler():
    '''
    Disable profiler from gathering data.
    '''
    cntk_py.disable_profiler()


