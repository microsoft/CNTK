from distutils.core import setup, Extension
from Cython.Build import cythonize
import os
import numpy

#os.environ['CFLAGS'] = "-I/usr/include/libusb-1.0"
#os.environ['LDFLAGS'] = "-lsetupapi"

CNTK_PATH = os.path.join(os.path.dirname(__file__), "..", "..")
SOURCE_PATH = os.path.join(CNTK_PATH, "Source")

ext_modules = [
    Extension(
           name="cython_cntk",                 

           sources=[r"cython_cntk.pyx"],

           libraries=[
               r"CNTKLibrary-2.0",
               r"Math"
               ],
           library_dirs=[os.path.join(CNTK_PATH, "x64", "Release_CpuOnly")],

           include_dirs=[
               #r"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include",
               os.path.join(SOURCE_PATH, "CNTKv2LibraryDll", "API"),
               os.path.join(SOURCE_PATH, "Math"),
               os.path.join(SOURCE_PATH, "Common", "Include"),
               numpy.get_include(),
               ],

           language="c++",             # generate C++ code

           extra_compile_args=["-DUNICODE", "/EHsc", "/DEBUG", "/Zi"],
           extra_link_args=[ "/DEBUG"],
            #cmdclass = {'build_ext': build_ext}
      )
    ]

setup(ext_modules = cythonize(ext_modules))
