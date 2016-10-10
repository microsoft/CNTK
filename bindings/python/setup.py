import sys
import os
import shutil
from glob import glob
import platform
from warnings import warn
from setuptools import setup, Extension, find_packages
import numpy

if sys.version_info.major < 3:
    print("Detected Python v2, which is not yet supported")
    sys.exit(1)

if shutil.which("swig") is None:
    print("Please install swig (>= 3.0.10) and include it in your path.\n")
    sys.exit(1)

IS_WINDOWS = platform.system() == 'Windows'

if IS_WINDOWS:
    if shutil.which("cl") is None:
        print("Compiler was not found in path. Please run this from a Visual Studio 2013 x64 Native Tools Command Prompt,\n"
              "e.g., by running the following command:\n"
              "  \"C:\\Program Files (x86)\\Microsoft Visual Studio 12.0\\VC\\vcvarsall\" amd64\n")
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
PROJ_LIB_PATH = os.path.join(os.path.dirname(__file__), "cntk", "libs")

if 'CNTK_LIB_PATH' in os.environ:
    CNTK_LIB_PATH = os.environ['CNTK_LIB_PATH']
else:
    if IS_WINDOWS:
        CNTK_LIB_PATH = os.path.join(CNTK_PATH, "x64", "Release")
    else:
        CNTK_LIB_PATH = os.path.join(
            CNTK_PATH, "build", "gpu", "release", "lib")

print("Using CNTK sources at '%s'" % os.path.abspath(CNTK_SOURCE_PATH))
print("Using CNTK libs at '%s'" % os.path.abspath(CNTK_LIB_PATH))


def lib_path(fn):
    return os.path.normpath(os.path.join(CNTK_LIB_PATH, fn))


def proj_lib_path(fn):
    return os.path.normpath(os.path.join(PROJ_LIB_PATH, fn))


def strip_path(fn):
    return os.path.split(fn)[1]


def strip_ext(fn):
    return os.path.splitext(fn)[0]

if IS_WINDOWS:
    libname_rt_ext = '.dll'

    link_libs = [strip_ext(strip_path(fn)) for fn in
                 glob(os.path.join(CNTK_LIB_PATH, '*.lib'))]
else:
    link_libs = [
        "cntklibrary-2.0",
        "cntkmath"
    ]
    libname_rt_ext = '.so'


rt_libs = [strip_path(fn) for fn in glob(os.path.join(CNTK_LIB_PATH,
                                                      '*' + libname_rt_ext))]

# copy over the libraries to the cntk base directory so that the rpath is
# correctly set
if os.path.exists(PROJ_LIB_PATH):
    shutil.rmtree(PROJ_LIB_PATH)

os.mkdir(PROJ_LIB_PATH)

for fn in rt_libs:
    src_file = lib_path(fn)
    tgt_file = proj_lib_path(fn)
    shutil.copy(src_file, tgt_file)

# For package_data we need to have names relative to the cntk module.
rt_libs = [os.path.join('libs', fn) for fn in rt_libs]

extra_compile_args = [
    "-DSWIG",
    "-DUNICODE"
]

if IS_WINDOWS:
    extra_compile_args += [
        "/EHsc",
        "/DEBUG",
        "/Zi",
        "/EHsc",
    ]
    runtime_library_dirs = []
else:
    extra_compile_args += [
        '--std=c++11',
    ]

    # Expecting the dependent libs (libcntklibrary-2.0.so, etc.) inside
    # site-packages/cntk/libs.
    runtime_library_dirs = ['$ORIGIN/cntk/libs']
    os.environ["CXX"] = "mpic++"

cntkV2LibraryInclude = os.path.join(CNTK_SOURCE_PATH, "CNTKv2LibraryDll", "API")

cntk_module = Extension(
    name="_cntk_py",

    sources = [os.path.join("cntk", "cntk_py.i")],
    swig_opts = ["-c++", "-D_MSC_VER", "-I" + cntkV2LibraryInclude],
    libraries = link_libs,
    library_dirs = [CNTK_LIB_PATH],

    runtime_library_dirs = runtime_library_dirs,

    include_dirs = [
        cntkV2LibraryInclude,
        os.path.join(CNTK_SOURCE_PATH, "Math"),
        os.path.join(CNTK_SOURCE_PATH, "Common", "Include"),
        numpy.get_include(),
    ],

    extra_compile_args=extra_compile_args,

    language="c++",
)

# Do not include examples
packages = [x for x in find_packages() if x.startswith('cntk') and not x.startswith('cntk.swig')]

package_data = { 'cntk': ['pytest.ini'] }

if IS_WINDOWS:
    # On Windows copy all runtime libs to the base folder of Python
    kwargs = dict(data_files = [('.', [ os.path.join('cntk', lib) for lib in rt_libs ])],
                  package_data = package_data)
else:
    # On Linux copy all runtime libs into the cntk/lib folder. 
    package_data['cntk'] += rt_libs
    kwargs = dict(package_data = package_data)

setup(name="cntk",
      version="2.0a3",
      url="http://cntk.ai",
      ext_modules=[cntk_module],
      packages=packages,
      # install_requires=[
      #  'numpy>=1.11',
      #  'scipy>=0.17'
      #],
      **kwargs)
