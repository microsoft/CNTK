REM set PYTHON_INCLUDE=c:\Anaconda3\include

REM Please change this
set PYTHON_LIB=c:\anaconda3\libs\python34.lib
rem set PYTHON_LIB=E:\WinPython-64bit-3.4.3.7\python-3.4.3.amd64\libs

REM Please change this
set SWIG=f:\swigwin-3.0.10\swig
rem set SWIG=E:\swigwin-3.0.10\swig

%SWIG% -c++ -python -D_MSC_VER -I..\..\..\..\Source\CNTKv2LibraryDll\API\ cntk_py.i
IF ERRORLEVEL 1 EXIT /B 1

copy cntk_py.py ..\
del cntk_py.py
