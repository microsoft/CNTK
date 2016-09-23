import os
import shutil
from glob import glob
import platform
from warnings import warn
from setuptools import setup, Extension, find_packages
import numpy

IS_WINDOWS = platform.system() == 'Windows'

CNTK_PATH = os.path.join(os.path.dirname(__file__), "..", "..")
CNTK_SOURCE_PATH = os.path.join(CNTK_PATH, "Source")
<<<<<<< HEAD
PROJ_LIB_PATH = os.path.join(os.path.dirname(__file__), "cntk", "libs")
=======
>>>>>>> 3ec0740... Further Linux improvements

if 'CNTK_LIB_PATH' in os.environ:
    CNTK_LIB_PATH = os.environ['CNTK_LIB_PATH']
else:
    CNTK_LIB_PATH = os.path.join(CNTK_PATH, "x64", "Release")

print("Using CNTK sources at '%s'"%os.path.abspath(CNTK_SOURCE_PATH))
print("Using CNTK libs at '%s'"%os.path.abspath(CNTK_LIB_PATH))

def lib_path(fn):
    return os.path.normpath(os.path.join(CNTK_LIB_PATH, fn))

def proj_lib_path(fn):
    return os.path.normpath(os.path.join(PROJ_LIB_PATH, fn))

def strip_path(fn):
    return os.path.split(fn)[1]

def strip_ext(fn):
    return os.path.splitext(fn)[0]

if IS_WINDOWS:
<<<<<<< HEAD
    libname_rt_ext = '.dll'

    link_libs = [strip_ext(strip_path(fn)) for fn in
            glob(os.path.join(CNTK_LIB_PATH, '*.lib'))]
else:
    link_libs=[
       "cntklibrary-2.0",
       "cntkmath"
=======
    libs=[
       os.path.join(CNTK_LIB_PATH, "CNTKLibrary-2.0"),
       os.path.join(CNTK_LIB_PATH, "Math"),
       os.path.join(CNTK_LIB_PATH, "BinaryReader"),
       os.path.join(CNTK_LIB_PATH, "DSSMReader"),
       os.path.join(CNTK_LIB_PATH, "EvalDll"),
       os.path.join(CNTK_LIB_PATH, "EvalWrapper"),
       os.path.join(CNTK_LIB_PATH, "nvml"),
       os.path.join(CNTK_LIB_PATH, "mkl_cntk_p"),
       os.path.join(CNTK_LIB_PATH, "libiomp5md"),
       os.path.join(CNTK_LIB_PATH, "cusparse64_75"),
       os.path.join(CNTK_LIB_PATH, "curand64_75"),
       os.path.join(CNTK_LIB_PATH, "cudnn64_5"),
       os.path.join(CNTK_LIB_PATH, "cudart64_75"),
       os.path.join(CNTK_LIB_PATH, "cublas64_75"),
       os.path.join(CNTK_LIB_PATH, "opencv_world310"),
    ]
else:
    libs=[
       os.path.join(CNTK_LIB_PATH, "libcntklibrary-2.0"),
       os.path.join(CNTK_LIB_PATH, "libcntkmath"),
       os.path.join(CNTK_LIB_PATH, "libeval"),
>>>>>>> 3ec0740... Further Linux improvements
    ]
    libname_rt_ext = '.so'


rt_libs = [strip_path(fn) for fn in glob(os.path.join(CNTK_LIB_PATH,
    '*'+libname_rt_ext))]

# copy over the libraries to the cntk base directory so that the rpath is correctly set
if os.path.exists(PROJ_LIB_PATH):
    shutil.rmtree(PROJ_LIB_PATH)

<<<<<<< HEAD
os.mkdir(PROJ_LIB_PATH)

for fn in rt_libs:
    src_file = lib_path(fn)
    tgt_file = proj_lib_path(fn)
    shutil.copy(src_file, tgt_file)

# For package_data we need to have names relative to the cntk module.
rt_libs = [os.path.join('libs', fn) for fn in rt_libs]

extra_compile_args = [
=======
libs += [
   os.path.join(CNTK_LIB_PATH, "CNTKTextFormatReader"),
   os.path.join(CNTK_LIB_PATH, "CompositeDataReader"),
   os.path.join(CNTK_LIB_PATH, "HTKDeserializers"),
   os.path.join(CNTK_LIB_PATH, "HTKMLFReader"),
   os.path.join(CNTK_LIB_PATH, "ImageReader"),
   os.path.join(CNTK_LIB_PATH, "LibSVMBinaryReader"),
   os.path.join(CNTK_LIB_PATH, "LMSequenceReader"),
   os.path.join(CNTK_LIB_PATH, "LUSequenceReader"),
   os.path.join(CNTK_LIB_PATH, "SparsePCReader"),
   os.path.join(CNTK_LIB_PATH, "UCIFastReader"),
   ]

extra_compile_args=[
>>>>>>> 3ec0740... Further Linux improvements
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

cntk_module = Extension(
        name="_cntk_py",

        sources=[os.path.join("cntk", "swig", "cntk_py_wrap.cxx")],

        libraries=link_libs,
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
packages = [x for x in find_packages() if x.startswith('cntk') and not x.startswith('cntk.swig')]

if IS_WINDOWS:
    # On Linux copy all runtime libs into the cntk/lib folder. 
    kwargs = dict(package_data = { 'cntk': rt_libs })
else:
    # On Windows copy all runtime libs to the base folder of Python
    kwargs = dict(data_files = [('.', [ os.path.join('cntk', lib) for lib in rt_libs ])])

setup(name="cntk", 
      version="2.0a2",
      url="http://cntk.ai",
      ext_modules = [cntk_module],  
      packages=packages,
      install_requires=[
        'numpy>=0.17',
        'scipy>=0.11'
      ],
      **kwargs
     )
