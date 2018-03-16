# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""
Check OS requirements before running CNTK for Python.
"""

import platform
import linecache
import warnings

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
