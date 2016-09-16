import setuptools
from distutils.core import setup, Extension
import distutils.sysconfig
import os
import platform
import numpy

IS_WINDOWS = platform.system() == 'Windows'

if 'CNTK_LIB_PATH' in os.environ:
    CNTK_LIB_PATH = os.environ['CNTK_LIB_PATH']
    CNTK_SOURCE_PATH = os.environ['CNTK_SOURCE_PATH']
else:
    CNTK_PATH = os.path.join(os.path.dirname(__file__), "..", "..")
    CNTK_SOURCE_PATH = os.path.join(CNTK_PATH, "Source")
    CNTK_LIB_PATH = os.path.join(CNTK_PATH, "x64", "Release")

print("Using CNTK sources at '%s'"%os.path.abspath(CNTK_SOURCE_PATH))
print("Using CNTK libs at '%s'"%os.path.abspath(CNTK_LIB_PATH))

#Todo: trim down the list of libs

if IS_WINDOWS:
    libs=[
       os.path.join(CNTK_LIB_PATH, "CNTKLibrary-2.0"),
    ]
else:
    libs=[
       os.path.join(CNTK_LIB_PATH, "libcntklibrary-2.0"),
    ]

libs += [
   os.path.join(CNTK_LIB_PATH, "CNTKLibrary-2.0"),
   os.path.join(CNTK_LIB_PATH, "BinaryReader"),
   os.path.join(CNTK_LIB_PATH, "CNTKTextFormatReader"),
   os.path.join(CNTK_LIB_PATH, "CompositeDataReader"),
   os.path.join(CNTK_LIB_PATH, "DSSMReader"),
   os.path.join(CNTK_LIB_PATH, "EvalDll"),
   os.path.join(CNTK_LIB_PATH, "EvalWrapper"),
   os.path.join(CNTK_LIB_PATH, "HTKDeserializers"),
   os.path.join(CNTK_LIB_PATH, "HTKMLFReader"),
   os.path.join(CNTK_LIB_PATH, "ImageReader"),
   os.path.join(CNTK_LIB_PATH, "LibSVMBinaryReader"),
   os.path.join(CNTK_LIB_PATH, "LMSequenceReader"),
   os.path.join(CNTK_LIB_PATH, "LUSequenceReader"),
   os.path.join(CNTK_LIB_PATH, "nvml"),
   os.path.join(CNTK_LIB_PATH, "SparsePCReader"),
   os.path.join(CNTK_LIB_PATH, "UCIFastReader"),
   os.path.join(CNTK_LIB_PATH, "Math"),
   os.path.join(CNTK_LIB_PATH, "mkl_cntk_p"),
   os.path.join(CNTK_LIB_PATH, "libiomp5md"),
   os.path.join(CNTK_LIB_PATH, "cusparse64_75"),
   os.path.join(CNTK_LIB_PATH, "curand64_75"),
   os.path.join(CNTK_LIB_PATH, "cudnn64_5"),
   os.path.join(CNTK_LIB_PATH, "cudart64_75"),
   os.path.join(CNTK_LIB_PATH, "cublas64_75"),
   os.path.join(CNTK_LIB_PATH, "opencv_world310"),
   ]

ext_modules = [
    Extension(
           name="_cntk_py",

           sources=[os.path.join("cntk", "swig", "cntk_py_wrap.cxx")],

           libraries=libs,
           library_dirs=[CNTK_LIB_PATH],

           include_dirs=[
               os.path.join(CNTK_SOURCE_PATH, "CNTKv2LibraryDll", "API"),
               os.path.join(CNTK_SOURCE_PATH, "Math"),
               os.path.join(CNTK_SOURCE_PATH, "Common", "Include"),
               numpy.get_include(),
               ],

           language="c++",

           extra_compile_args=[
               "-DSWIG",
               "-DUNICODE",
               "/EHsc"
               "/DEBUG",
               "/Zi" if IS_WINDOWS else '',
               "/EHsc" if IS_WINDOWS else '',
               '--std=c++11' if not IS_WINDOWS else '',
               ],
           #extra_link_args=[ "/DEBUG"],
      )
    ]

#import pdb;pdb.set_trace()
#TODO: do not copy the dlls to the root, try to find a better location that is accessible
if IS_WINDOWS:
    data_files = [('.\\', [ lib + ".dll" for lib in libs ])] 
else:
    data_files = [('./', [ lib + ".so" for lib in libs ])] 
    #os.environ["CC"] = "/usr/bin/gcc-4.8" 
    #os.environ["CXX"] = "/usr/bin/gcc-4.8"
    os.environ["CC"] = "/usr/local/openmpi-1.10.1/bin/mpic++" 
    os.environ["CXX"] = "/usr/local/openmpi-1.10.1/bin/mpic++"

packages = [x for x in setuptools.find_packages() if x.startswith('cntk')]

import distutils.debug
distutils.debug.DEBUG = True
setup(name="cntk", 
    ext_modules = ext_modules,  
    data_files = data_files,
    packages=packages)
