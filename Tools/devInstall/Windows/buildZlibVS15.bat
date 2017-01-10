@REM Copyright (c) Microsoft. All rights reserved.
@REM Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
@REM
@REM batch script to build the compression library used by the CNTK image reader

@echo off

if /I "%CMDCMDLINE%" neq ""%COMSPEC%" " (
    @echo.
    @echo Please execute this script from inside a regular Windows command prompt.
    @echo.
    exit /b 0
)

setlocal

if "%~1"=="" goto HELP
if "%~1"=="-?" goto HELP
if /I "%~1"=="-h" goto HELP
if /I "%~1"=="-help" goto HELP
if "%~2"=="" goto HELP
if "%~3"=="" goto HELP
if not "%~4"=="" goto HELP

set LIBZIPSOURCEDIR=%~f1
set ZLIBSOURCEDIR=%~f2
set TARGETDIR=%~f3

if "%LIBZIPSOURCEDIR:~-1%"=="\" set LIBZIPSOURCEDIR=%LIBZIPSOURCEDIR:~,-1%
if "%ZLIBSOURCEDIR:~-1%"=="\" set ZLIBSOURCEDIR=%ZLIBSOURCEDIR:~,-1%
if "%TARGETDIR:~-1%"=="\" set TARGETDIR=%TARGETDIR:~,-1%

if not exist "%LIBZIPSOURCEDIR%\CMakeLists.txt" (
  @echo Error: "%LIBZIPSOURCEDIR%" not a valid LibZib directory 
  goto :FIN
)
if not exist "%ZLIBSOURCEDIR%\CMakeLists.txt" (
  @echo Error: "%ZLIBSOURCEDIR%" not a valid ZLib directory 
  goto :FIN
)
where -q cmake.exe
if errorlevel 1 (
  @echo Error: CMAKE.EXE not found in PATH!
  goto FIN
)

if not defined VS140COMNTOOLS ( 
  @echo Environment variable VS140COMNTOOLS not defined.
  @echo Make sure Visual Studion 2015 Update 3 is installed.
  goto FIN
)
set VCDIRECTORY=%VS140COMNTOOLS%
if "%VCDIRECTORY:~-1%"=="\" set VCDIRECTORY=%VCDIRECTORY:~,-1%

if not exist "%VCDIRECTORY%\..\..\VC\vcvarsall.bat" (
  echo Error: "%VCDIRECTORY%\..\..\VC\vcvarsall.bat" not found. 
  echo Make sure you have installed Visual Studion 2015 Update 3 correctly.  
  goto FIN
)

@set CMAKEGEN="Visual Studio 14 2015 Win64"

@echo.
@echo This will build cthe compression library used by the CNTK image reader
@echo ----------------------------------------------------------------------
@echo The configured settings for the batch file:
@echo    Visual Studio directory: %VCDIRECTORY%
@echo    CMake Generator: %CMAKEGEN%
@echo    LibZip source directory: %LIBZIPSOURCEDIR%
@echo    Zlib source directory: %ZLIBSOURCEDIR%
@echo    Zlib-VS15 target directory: %TARGETDIR%
@echo.

@pause 

call "%VCDIRECTORY%\..\..\VC\vcvarsall.bat" amd64 

@pushd "%ZLIBSOURCEDIR%"
@mkdir build
@cd build
@cmake .. -G%CMAKEGEN% -DCMAKE_INSTALL_PREFIX="%TARGETDIR%"
@msbuild /P:Configuration=Release INSTALL.vcxproj
@popd

@pushd "%LIBZIPSOURCEDIR%"
@md build
@cd build
@cmake .. -G%CMAKEGEN% -DCMAKE_INSTALL_PREFIX="%TARGETDIR%"
@msbuild libzip.sln /t:zip /P:Configuration=Release
@cmake -DBUILD_TYPE=Release -P cmake_install.cmake
@popd

goto FIN

:HELP
@echo.
@echo Use this script to build the the compression library used by the CNTK image reader
@echo The script requires three parameter
@echo   Parameter 1: The complete path to the LibZip source directory 
@echo                e.g C:\local\src\libzip-1.1.3
@echo   Parameter 1: The complete path to the ZLib source directory 
@echo                e.g C:\local\src\zlib-1.2.8
@echo   Parameter 2: The target path for the created binaries
@echo                e.g C:\local\zlib-vs15
@echo.
goto FIN

:FIN
endlocal

REM vim:set expandtab shiftwidth=2 tabstop=2:
