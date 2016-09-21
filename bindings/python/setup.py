import os
import shutil
import platform
from warnings import warn
from setuptools import setup, Extension, find_packages
import numpy

IS_WINDOWS = platform.system() == 'Windows'

CNTK_PATH = os.path.join(os.path.dirname(__file__), "..", "..")
CNTK_SOURCE_PATH = os.path.join(CNTK_PATH, "Source")
PROJ_LIB_PATH = os.path.join(os.path.dirname(__file__), "cntk", "libs")

if 'CNTK_LIB_PATH' in os.environ:
    CNTK_LIB_PATH = os.environ['CNTK_LIB_PATH']
else:
    CNTK_LIB_PATH = os.path.join(CNTK_PATH, "x64", "Release")

print("Using CNTK sources at '%s'"%os.path.abspath(CNTK_SOURCE_PATH))
print("Using CNTK libs at '%s'"%os.path.abspath(CNTK_LIB_PATH))

#Todo: trim down the list of libs

def lib_path(fn):
    return os.path.normpath(os.path.join(CNTK_LIB_PATH, fn))
    #return os.path.join(CNTK_LIB_PATH, fn)

def proj_lib_path(fn):
    return os.path.normpath(os.path.join(PROJ_LIB_PATH, fn))

dep_libs_names = ["CNTKTextFormatReader", "CompositeDataReader", "HTKDeserializers", "HTKMLFReader", "ImageReader", "LibSVMBinaryReader", "LMSequenceReader", "LUSequenceReader", "SparsePCReader",
"UCIFastReader"]

if IS_WINDOWS:
    libs=[
       "CNTKLibrary-2.0",
       "Math",
       "BinaryReader",
       "DSSMReader",
       "EvalDll",
       "EvalWrapper",
       "nvml",
       "mkl_cntk_p",
       "libiomp5md",
       "cusparse64_75",
       "curand64_75",
       "cudnn64_5",
       "cudart64_75",
       "cublas64_75",
       "opencv_world310",
    ]
    libname_prefix = ''
    libname_ext = '.dll'
else:
    libs=[
       "cntklibrary-2.0",
       "cntkmath"
    ]
    libname_prefix = 'lib'
    libname_ext = '.so'

dep_libs_names = [l+libname_ext for l in dep_libs_names + [libname_prefix+l for l in libs]]

# copy over the libraries to the cntk base directory so that the rpath is correctly set
if os.path.exists(PROJ_LIB_PATH):
    shutil.rmtree(PROJ_LIB_PATH)

os.mkdir(PROJ_LIB_PATH)

for fn in dep_libs_names:
    src_file = lib_path(fn)
    tgt_file = proj_lib_path(fn)
    if os.path.exists(src_file):
        shutil.copy(src_file, tgt_file)
    else:
        warn("Didn't find library file %s"%src_file)

# for package_data we need to have names relative to the cntk module
dep_libs = [os.path.join('libs', fn) for fn in dep_libs_names]

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
    #TODO: do not copy the dlls to the root, try to find a better location that is accessible
else:
    extra_compile_args += [
	'--std=c++11',
    ]
    # expecting the dependent libs (libcntklibrary-2.0.so, etc.) inside site-package/cntk
    runtime_library_dirs = ['$ORIGIN/cntk/libs']
    os.environ["CXX"] = "mpic++"

cntk_module = Extension(
           name="_cntk_py",

           sources=[os.path.join("cntk", "swig", "cntk_py_wrap.cxx")],

           libraries=libs,
           library_dirs=[CNTK_LIB_PATH],

           runtime_library_dirs=runtime_library_dirs,

           include_dirs=[
               os.path.join(CNTK_SOURCE_PATH, "CNTKv2LibraryDll", "API"),
               os.path.join(CNTK_SOURCE_PATH, "Math"),
               os.path.join(CNTK_SOURCE_PATH, "Common", "Include"),
               numpy.get_include(),
               ],

            extra_compile_args = extra_compile_args,

           language="c++",
      )

# do not include tests and examples
packages = [x for x in find_packages() if x.startswith('cntk')]

setup(name="cntk", 
      version="2.0.a2",
      url="http://cntk.ai",
      ext_modules = [cntk_module],  
      packages=packages,
      package_data = { 'cntk': dep_libs },
      install_requires=[
        'numpy>=0.17',
        'scipy>=0.11'
      ],
     )
