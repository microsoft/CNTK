from distutils.core import setup, Extension
import os
import numpy

CNTK_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..")
CNTK_SOURCE_PATH = os.path.join(CNTK_PATH, "Source")
CNTK_LIB_PATH = os.path.join(CNTK_PATH, "x64", "Release_CpuOnly")
print("Using CNTK libs at '%s'"%os.path.abspath(CNTK_LIB_PATH))
#CNTK_LIB_PATH = os.path.join(CNTK_PATH, "x64", "Debug_CpuOnly")

print( os.path.join(CNTK_SOURCE_PATH, "CNTKv2LibraryDll", "API"))

ext_modules = [
    Extension(
           name="_cntk_py",

           sources=[os.path.join("swig", "cntk_py_wrap.cxx")],

           libraries=[
               os.path.join(CNTK_LIB_PATH, "CNTKLibrary-2.0"),
               os.path.join(CNTK_LIB_PATH, "Math")
               ],
           library_dirs=[CNTK_LIB_PATH],

           include_dirs=[
               os.path.join(CNTK_SOURCE_PATH, "CNTKv2LibraryDll", "API"),
               os.path.join(CNTK_SOURCE_PATH, "Math"),
               os.path.join(CNTK_SOURCE_PATH, "Common", "Include"),
               numpy.get_include(),
               ],

           language="c++",

           extra_compile_args=[
               "-DUNICODE", 
               "/EHsc", 
               "/DEBUG", # TODO remove for release
               "/Zi",     # TODO remove for release
               "/Od"     # TODO remove for release
               ],
           extra_link_args=[ "/DEBUG"],
      )
    ]

setup(ext_modules = ext_modules)
