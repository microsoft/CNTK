import setuptools
from distutils.core import setup, Extension
import distutils.sysconfig
import os
import numpy


CNTK_PATH = os.path.join(os.path.dirname(__file__), "..", "..")
CNTK_SOURCE_PATH = os.path.join(CNTK_PATH, "Source")
#CNTK_LIB_PATH = os.path.join(CNTK_PATH, "x64", "Release_CpuOnly")
CNTK_LIB_PATH = os.path.join(CNTK_PATH, "x64", "Debug_CpuOnly")
print("Using CNTK libs at '%s'"%os.path.abspath(CNTK_LIB_PATH))

print( os.path.join(CNTK_SOURCE_PATH, "CNTKv2LibraryDll", "API"))

#Todo: trim down the list of libs
libs=[
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
               ],
           extra_link_args=[ "/DEBUG"],
      )
    ]

#TODO: do not copy the dlls to the root, try to find a better location that is accessible
setup(name="cntk", ext_modules = ext_modules,  data_files = [('.\\', [ lib + ".dll" for lib in libs ])], packages=[x for x in setuptools.find_packages() if x.startswith('cntk')])
