from distutils.core import setup, Extension
import os
import numpy

CNTK_PATH = os.path.join(os.path.dirname(__file__), "..", "..")
CNTK_SOURCE_PATH = os.path.join(CNTK_PATH, "Source")
CNTK_LIB_PATH = os.path.join(CNTK_PATH, "x64", "Release_CpuOnly")

print( os.path.join(CNTK_SOURCE_PATH, "CNTKv2LibraryDll", "API"))

ext_modules = [
    Extension(
           # name has to start with underscore, since SWIG will create a
           # non-underscore .py file, which will import this one
           name="_swig_cntk",

           sources=[r"swig_cntk_wrap.cxx"],

           libraries=[
               os.path.join(CNTK_LIB_PATH, "CNTKLibrary-2.0"),
               os.path.join(CNTK_LIB_PATH, "Math")
               ],
           library_dirs=[CNTK_LIB_PATH],

           include_dirs=[
               #r"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include",
               os.path.join(CNTK_SOURCE_PATH, "CNTKv2LibraryDll", "API"),
               os.path.join(CNTK_SOURCE_PATH, "Math"),
               os.path.join(CNTK_SOURCE_PATH, "Common", "Include"),
               numpy.get_include(),
               ],

           language="c++",             # generate C++ code

           extra_compile_args=["-DUNICODE", "/EHsc", "/DEBUG", "/Zi"],
           extra_link_args=[ "/DEBUG"],
            #cmdclass = {'build_ext': build_ext}
      )
    ]

setup(ext_modules = ext_modules)
