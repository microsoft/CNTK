@REM Copyright (c) Microsoft. All rights reserved.
@REM Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
@REM
@REM batch script to build mkl-dnn library for CNTK

@echo off
if /I "%CMDCMDLINE%" neq ""%COMSPEC%" " (
    echo.
    echo Please execute this script from inside a regular Windows command prompt.
    echo.
    exit /b 0
)
setlocal
if "%~1"=="" goto HELP
if "%~1"=="-?" goto HELP
if /I "%~1"=="-h" goto HELP
if /I "%~1"=="-help" goto HELP
if "%~2"=="" goto HELP
if not "%~3"=="" goto HELP

set SOURCEDIR=%~f1
set TARGETDIR=%~f2

if "%SOURCEDIR:~-1%"=="\" set SOURCEDIR=%SOURCEDIR:~,-1%
if "%TARGETDIR:~-1%"=="\" set TARGETDIR=%TARGETDIR:~,-1%

where -q cmake.exe
if errorlevel 1 (
  echo Error: CMAKE.EXE not found in PATH!
  goto FIN
)

if not defined VS2017INSTALLDIR (
  echo Environment variable VS2017INSTALLDIR not defined.
  echo Make sure Visual Studion 2017 is installed.
  goto FIN
)
set VCVARSALL=%VS2017INSTALLDIR%\VC\Auxiliary\Build\vcvarsall.bat

if not exist "%VCVARSALL%" (
  echo Error: "%VCVARSALL%" not found. 
  echo Make sure you have installed Visual Studion 2017 correctly.  
  goto FIN
)

if exist "%TARGETDIR%\include\mkl_version.h" (
  set MKLROOT=%TARGETDIR%
  echo Found mklml in %TARGETDIR%
) else (
  if not defined MKLML_PATH (
    echo Environment variable MKLML_PATH not defined.
    echo MKL-DNN will be built without MKLML.
  )
  set MKLROOT=%MKLML_PATH%
)

echo.
echo This will build mkl-dnn for CNTK using Visual Studio 2017
echo ----------------------------------------------------------
echo The configured settings for the batch file:
echo    Visual Studio directory: %VS2017INSTALLDIR%
echo    mkl-dnn source directory: %SOURCEDIR%
echo    mkl-dnn target directory: %TARGETDIR%
echo    mklml directory: %MKLROOT%
echo.

pause 

call "%VCVARSALL%" amd64 -vcvars_ver=14.11

pushd "%SOURCEDIR%"

cd cmake
cmake -G "Visual Studio 15 Win64" ..
msbuild "Intel(R) MKL-DNN.sln" /t:Rebuild /p:Configuration=Release /m

popd

if not exist "%TARGETDIR%" (
    md "%TARGETDIR%"
)
pushd "%TARGETDIR%"
if not exist include (md include)
copy "%SOURCEDIR%\include\*" include
if not exist lib (md lib)
copy "%SOURCEDIR%\cmake\src\Release\*.lib" lib
copy "%SOURCEDIR%\cmake\src\Release\*.dll" lib
popd

goto FIN

:HELP
echo.
echo Use this script to build the mkl-dnn library for CNTK.
echo The script requires two parameter
echo   Parameter 1: The complete path to the mkl-dnn source directory 
echo                e.g. C:\local\src\mkl-dnn-0.14
echo   Parameter 2: The target path for the created binaries
echo                e.g. C:\local\mkl-dnn-0.12, or c:\local\mklml\mklml_win_2018.0.1.20171227
echo Note: mkl-dnn could be built with or without MKLML (a trimmed version of Intel MKL).
echo       If the target path is mklml path, or if environment variable MKLML_PATH is defined,
echo       this script builds mkl-dnn using mklml. Otherwise, it builds mkl-dnn without mklml.
echo.
goto FIN

:FIN
endlocal

REM vim:set expandtab shiftwidth=2 tabstop=2:
