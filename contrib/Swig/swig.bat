REM set PYTHON_INCLUDE=c:\Anaconda3\include
set PYTHON_LIB=c:\anaconda3\libs\python34.lib

REM Please change this
set SWIG=c:\blis\PyCNTK\swigwin-3.0.10\swig
REM set SWIG=f:\swigwin-3.0.10\swig

%SWIG% -c++ -python -D_MSC_VER -I..\..\Source\CNTKv2LibraryDll\API\ swig_cntk.i 

