@echo off
setlocal
cd "%~dp0"
set USAGE=x
REM Please specify absolute path to the SWIG executable in %%SWIG%%, or put it in path.
if not defined SWIG (
  where /q swig.exe
  if errorlevel 1 (
    echo %USAGE%
    exit /b 1
  )
  set SWIG=swig.exe
) else if not exist "%SWIG%" (
    echo %USAGE%
    exit /b 1
)

@REM Sanity check.
@REM TODO really do a version check
"%SWIG%" -version 1> NUL: 2> NUL:
if errorlevel 1 (
  echo Cannot determine SWIG version.
  exit /b 1
)

set CMD="%SWIG%" -c++ -python -D_MSC_VER -I..\..\..\..\Source\CNTKv2LibraryDll\API\ cntk_py.i
%CMD%
if errorlevel 1 (
  echo Command failed to run: %CMD%
  exit /b 1
)

move cntk_py.py ..
