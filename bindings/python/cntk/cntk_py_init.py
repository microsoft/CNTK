# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""
Check OS requirements before running CNTK for Python.
"""
import sys
import os
import platform
import linecache
import warnings
from subprocess import call

def cntk_check_distro_info():
    __my_distro__ = ''
    __my_distro_ver__ = ''
    __my_system__ = platform.system().lower()
    __my_arch__ = platform.architecture()[0].lower()

    __OS_RELEASE_FILE__ = '/etc/os-release'
    __LSB_RELEASE_FILE__ = '/etc/lsb-release'

    if __my_arch__ != '64bit':
        warnings.warn('Unsupported architecture (%s). CNTK supports 64bit architecture, only.' % __my_arch__)

    if __my_system__ == 'windows':
        __my_distro__ = __my_system__
        __my_distro_ver__ = platform.release().lower()

        if __my_distro_ver__ != '10':
            warnings.warn('Unsupported Windows version (%s). CNTK supports Windows 10 and above, only.' % __my_distro_ver__)
    elif __my_system__ == 'linux':
        ''' Although the 'platform' python module for getting Distro information works well on standard OS images running on real
        hardware, it is not acurate when running on Azure VMs, Git Bash, Cygwin, etc. The returned values for release and version
        are unpredictable for virtualized or emulated environments.

        /etc/os-release and /etc/lsb_release files, on the other hand, are guaranteed to exist and have standard values in all
        OSes supported by CNTK. The former is the current standard file to check OS info and the latter is its antecessor.
        '''
        # Newer systems have /etc/os-release with relevant distro info
        __my_distro__ = linecache.getline(__OS_RELEASE_FILE__, 3)[3:-1]
        __my_distro_ver__ = linecache.getline(__OS_RELEASE_FILE__, 6)[12:-2]

        # Older systems may have /etc/os-release instead
        if not __my_distro__:
            __my_distro__ = linecache.getline(__LSB_RELEASE_FILE__, 1)[11:-1]
            __my_distro_ver__ = linecache.getline(__LSB_RELEASE_FILE__, 2)[16:-1]

        # Instead of trying to parse distro specific files,
        # warn the user CNTK may not work out of the box
        __my_distro__ = __my_distro__.lower()
        __my_distro_ver__ = __my_distro_ver__.lower()

        if __my_distro__ != 'ubuntu' or __my_distro_ver__ != '16.04':
            warnings.warn('Unsupported Linux distribution (%s-%s). CNTK supports Ubuntu 16.04 and above, only.' % (__my_distro__, __my_distro_ver__))
    else:
        warnings.warn('Unsupported platform (%s). CNTK supports Linux and Windows platforms, only.' % __my_system__)

def cntk_check_libs():
    WARNING_MSG='\n\n'+('#'*48)
    WARNING_MSG+=' Missing optional dependency (%s) '
    WARNING_MSG+=('#'*48)
    WARNING_MSG+='\n   CNTK may crash if the component that depends on those dependencies is loaded.'
    WARNING_MSG+='\n   Visit %s for more information.'
    WARNING_MSG+='\n'+('#'*140)+'\n'
    WARNING_MSG_GPU_ONLY=WARNING_MSG[:-1]
    WARNING_MSG_GPU_ONLY+='\nIf you intend to use CNTK without GPU support, you can ignore the (likely) GPU-specific warning!'
    WARNING_MSG_GPU_ONLY+='\n'+('#'*140)+'\n'

    devnull = open(os.devnull, 'w')
    __my_system__ = platform.system().lower()
    if __my_system__ == 'windows':
        if call(['where', 'libiomp5md*.dll'], stdout=devnull, stderr=devnull) != 0 or \
          call(['where', 'mklml*.dll'], stdout=devnull, stderr=devnull) != 0:
            warnings.warn(WARNING_MSG % ('    MKL     ', 'https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-Windows-Python#mkl'))
        if call(['where', 'cudnn*.dll'], stdout=devnull, stderr=devnull) != 0 or \
          call(['where', 'nvml*.dll'], stdout=devnull, stderr=devnull) != 0 or \
          call(['where', 'nvml*.dll'], stdout=devnull, stderr=devnull) != 0 or \
          call(['where', 'cublas*.dll'], stdout=devnull, stderr=devnull) != 0 or \
          call(['where', 'cudart*.dll'], stdout=devnull, stderr=devnull) != 0 or \
          call(['where', 'curand*.dll'], stdout=devnull, stderr=devnull) != 0 or \
          call(['where', 'cusparse*.dll'], stdout=devnull, stderr=devnull) != 0:
            warnings.warn(WARNING_MSG_GPU_ONLY % ('GPU-Specific', 'https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-Windows-Python#optional-gpu-specific-packages'))
        if call(['where', 'opencv_world*.dll'], stdout=devnull, stderr=devnull) != 0:
            warnings.warn(WARNING_MSG % ('   OpenCV   ', 'https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-Windows-Python#optional-opencv'))
    elif __my_system__ == 'linux':
        if call('ldconfig -N -v $(sed "s/:/ /g" <<< $LD_LIBRARY_PATH) | grep libmklml_intel*.so*', shell=True, executable='/bin/bash', stdout=devnull, stderr=devnull) != 0 or \
          call('ldconfig -N -v $(sed "s/:/ /g" <<< $LD_LIBRARY_PATH) | grep libiomp5*.so*', shell=True, executable='/bin/bash', stdout=devnull, stderr=devnull) != 0:
            warnings.warn(WARNING_MSG % ('    MKL     ', 'https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-Linux-Python#mkl'))
        if call('ldconfig -N -v $(sed "s/:/ /g" <<< $LD_LIBRARY_PATH) | grep libcudart*.so*', shell=True, executable='/bin/bash', stdout=devnull, stderr=devnull) != 0 or \
          call('ldconfig -N -v $(sed "s/:/ /g" <<< $LD_LIBRARY_PATH) | grep libcublas*.so*', shell=True, executable='/bin/bash', stdout=devnull, stderr=devnull) != 0 or \
          call('ldconfig -N -v $(sed "s/:/ /g" <<< $LD_LIBRARY_PATH) | grep libcurand*.so*', shell=True, executable='/bin/bash', stdout=devnull, stderr=devnull) != 0 or \
          call('ldconfig -N -v $(sed "s/:/ /g" <<< $LD_LIBRARY_PATH) | grep libcusparse*.so*', shell=True, executable='/bin/bash', stdout=devnull, stderr=devnull) != 0 or \
          call('ldconfig -N -v $(sed "s/:/ /g" <<< $LD_LIBRARY_PATH) | grep libcuda*.so*', shell=True, executable='/bin/bash', stdout=devnull, stderr=devnull) != 0 or \
          call('ldconfig -N -v $(sed "s/:/ /g" <<< $LD_LIBRARY_PATH) | grep libnvidia-ml*.so*', shell=True, executable='/bin/bash', stdout=devnull, stderr=devnull) != 0 or \
          call('ldconfig -N -v $(sed "s/:/ /g" <<< $LD_LIBRARY_PATH) | grep libcudnn*.so*', shell=True, executable='/bin/bash', stdout=devnull, stderr=devnull) != 0:
            warnings.warn(WARNING_MSG_GPU_ONLY % ('GPU-Specific', 'https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-Linux-Python#optional-gpu-specific-packages'))
        if call('ldconfig -N -v $(sed "s/:/ /g" <<< $LD_LIBRARY_PATH) | grep libopencv_core*.so*', shell=True, executable='/bin/bash', stdout=devnull, stderr=devnull) != 0 or \
          call('ldconfig -N -v $(sed "s/:/ /g" <<< $LD_LIBRARY_PATH) | grep libopencv_imgproc*.so*', shell=True, executable='/bin/bash', stdout=devnull, stderr=devnull) != 0 or \
          call('ldconfig -N -v $(sed "s/:/ /g" <<< $LD_LIBRARY_PATH) | grep libopencv_imgcodecs*.so*', shell=True, executable='/bin/bash', stdout=devnull, stderr=devnull) != 0:
            warnings.warn(WARNING_MSG % ('   OpenCV   ', 'https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-Linux-Python#optional-opencv'))
    devnull.close()