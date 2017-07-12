# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import sys
import os
import shutil
from glob import glob
import platform
from setuptools import setup, Extension, find_packages
import numpy

IS_WINDOWS = platform.system() == 'Windows'

IS_PY2 = sys.version_info.major == 2

# TODO should handle swig path specified via build_ext --swig-path
if os.system('swig -version 1>%s 2>%s' % (os.devnull, os.devnull)) != 0:
    print("Please install swig (>= 3.0.10) and include it in your path.\n")
    sys.exit(1)

if IS_WINDOWS:
    if os.system('cl 1>%s 2>%s' % (os.devnull, os.devnull)) != 0:
        print("Compiler was not found in path.\n"
              "Make sure you installed the C++ tools during Visual Studio 2015 install and \n"
              "run vcvarsall.bat from a DOS command prompt:\n"
              "  \"C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\vcvarsall\" amd64\n")
        sys.exit(1)

    try:
        assert(os.environ["MSSdk"] == "1");
        assert(os.environ["DISTUTILS_USE_SDK"] == "1");
    except (KeyError, AssertionError) as e:
        print("Please set the environment variables MSSdk and DISTUTILS_USE_SDK to 1:\n"
              "  set MSSdk=1\n"
              "  set DISTUTILS_USE_SDK=1\n")
        sys.exit(1)

CNTK_PATH = os.path.join(os.path.dirname(__file__), "..", "..")
CNTK_SOURCE_PATH = os.path.join(CNTK_PATH, "Source")

# When called through MSBuild / Makefile, environment variables are set up to
# to identify the shared library directory and list of required shared libraries.
# Otherwise (setup.py called directly), assume it's a GPU SKU build with
# default directories.

if 'CNTK_LIB_PATH' in os.environ:
    CNTK_LIB_PATH = os.environ['CNTK_LIB_PATH']
else:
    # Assumes GPU SKU is being built
    if IS_WINDOWS:
        CNTK_LIB_PATH = os.path.join(CNTK_PATH, "x64", "Release")
    else:
        CNTK_LIB_PATH = os.path.join(
            CNTK_PATH, "build", "gpu", "release", "lib")

shared_library_glob = '*.dll' if IS_WINDOWS else '*.so*'

if 'CNTK_LIBRARIES' in os.environ:
    CNTK_SHARED_LIBARIES = [os.path.join(CNTK_LIB_PATH, fn) for fn in
        os.environ['CNTK_LIBRARIES'].split(';' if IS_WINDOWS else None)]
else:
    CNTK_SHARED_LIBARIES = glob(os.path.join(CNTK_LIB_PATH, shared_library_glob))

if 'CNTK_EXTRA_LIBRARIES' in os.environ:
    CNTK_SHARED_LIBARIES += os.environ['CNTK_EXTRA_LIBRARIES'].split(';' if IS_WINDOWS else None)

# Copy all libraries into a package-local folder (root on Windows, libs on Linux)
package_name = 'cntk'
relative_shared_library_path = [] if IS_WINDOWS else ['libs']
shared_library_path = os.path.join(os.path.dirname(__file__), package_name, *relative_shared_library_path)

if not os.path.isdir(shared_library_path):
    os.mkdir(shared_library_path)

for fn in glob(os.path.join(shared_library_path, shared_library_glob)):
    os.remove(fn)

for fn in CNTK_SHARED_LIBARIES:
    shutil.copy(fn, shared_library_path)

# Extensions module
extra_compile_args = [
    "-DSWIG",
    "-DUNICODE"
]

if IS_WINDOWS:
    extra_compile_args += [
        "/EHsc",
        "/DEBUG",
        "/Zi",
        "/WX"
    ]
    extra_link_args = ['/DEBUG']
    runtime_library_dirs = []
else:
    extra_compile_args += [
        '--std=c++11',
    ]
    extra_link_args = []

    runtime_library_dirs = [os.path.join('$ORIGIN', *relative_shared_library_path)]
    os.environ["CXX"] = "mpic++"

cntkV2LibraryInclude = os.path.join(CNTK_SOURCE_PATH, "CNTKv2LibraryDll", "API")
cntkBindingCommon = os.path.join(CNTK_PATH, "bindings", "common")

cntk_module = Extension(
    name=package_name + '._cntk_py',

    sources = [os.path.join(package_name, "cntk_py.i")],
    swig_opts = ["-c++", "-D_MSC_VER", "-I" + cntkV2LibraryInclude, "-I" + cntkBindingCommon, "-Werror" ],
    libraries = ["Cntk.Core-" + os.environ['CNTK_COMPONENT_VERSION']],

    library_dirs = [CNTK_LIB_PATH],

    runtime_library_dirs = runtime_library_dirs,

    include_dirs = [
        cntkV2LibraryInclude,
        os.path.join(CNTK_SOURCE_PATH, "Math"),
        os.path.join(CNTK_SOURCE_PATH, "Common", "Include"),
        numpy.get_include(),
    ],

    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
    depends = [os.path.join(CNTK_SOURCE_PATH, "Common", "Include", "ExceptionWithCallStack.h")] +
        [os.path.join(cntkBindingCommon, f) for f in ["CNTKExceptionHandling.i", "CNTKValueExtend.i", "CNTKWarnFilters.i"]] +
        [os.path.join(cntkV2LibraryInclude, f) for f in ["CNTKLibraryInternals.h", "CNTKLibrary.h"]],
)

packages = [x for x in find_packages() if x.startswith(package_name)]

package_data = { package_name: ['pytest.ini', os.path.join(*(relative_shared_library_path + [shared_library_glob]))] }

install_requires = [
    'numpy>=1.11',
    'scipy>=0.17'
]

if IS_PY2:
    install_requires.append('enum34>=1.1.6')

setup(name=package_name,
      version="2.0",
      url='https://docs.microsoft.com/cognitive-toolkit/',
      ext_modules=[cntk_module],
      packages=packages,
      install_requires=install_requires,
      package_data=package_data)
