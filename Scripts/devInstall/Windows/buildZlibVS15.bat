@REM Copyright (c) Microsoft. All rights reserved.
@REM Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
@REM
@REM batch script to build image compression libraries for CNTK

@echo off
if /I "%CMDCMDLINE%" neq ""%COMSPEC%" " (
    @echo.
    @echo Please execute this script from inside a regular Windows command prompt.
    @echo.
    exit /b 0
)

if "%1"=="" ( goto HELP )
if "%1"=="-?" ( goto HELP )
if /I "%1"=="-h" ( goto HELP )
if /I "%1"=="-help" ( goto HELP )
if "%2"=="" ( goto HELP )
if "%3"=="" ( goto HELP )
if not "%4"=="" ( goto HELP )

SET LIBZIPSOURCEDIR=%~f1
set ZLIBSOURCEDIR=%~f2
SET TARGETDIR=%~f3

IF "%LIBZIPSOURCEDIR:~-1%"=="\" SET LIBZIPSOURCEDIR=%LIBZIPSOURCEDIR:~,-1%
IF "%ZLIBSOURCEDIR:~-1%"=="\" SET ZLIBSOURCEDIR=%ZLIBSOURCEDIR:~,-1%
IF "%TARGETDIR:~-1%"=="\" SET TARGETDIR=%TARGETDIR:~,-1%

if not exist "%LIBZIPSOURCEDIR%\CMakeLists.txt" (
  @echo Error: "%LIBZIPSOURCEDIR%" not a valid LibZib directory 
  goto :FIN
)
if not exist "%ZLIBSOURCEDIR%\CMakeLists.txt" (
  @echo Error: "%ZLIBSOURCEDIR%" not a valid ZLib directory 
  goto :FIN
)

SET VCDIRECTORY=%VS140COMNTOOLS%
if "%VCDIRECTORY%"=="" ( 
  @echo Environment variable VS140COMNTOOLS not defined.
  @echo Make sure Visual Studion 2015 Update 3 is installed.
  goto FIN
)

IF "%VCDIRECTORY:~-1%"=="\" SET VCDIRECTORY=%VCDIRECTORY:~,-1%

if not exist "%VCDIRECTORY%\..\..\VC\vcvarsall.bat" (
  echo Error: "%VCDIRECTORY%\..\..\VC\vcvarsall.bat" not found. 
  echo Make sure you have installed Visual Studion 2015 Update 3 correctly.  
  goto FIN
)

@SET CMAKEGEN="Visual Studio 14 2015 Win64"

@echo.
@echo This will build compression libraries for CNTK using Visual Studio 2015
@echo -----------------------------------------------------------------------
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

setx ZLIB_PATH %TARGETDIR%
set ZLIB_PATH=%TARGETDIR%

goto FIN

:HELP
@echo.
@echo Use this script to build the image compression libraries for CNTK.
@echo The script requires three parameter
@echo   Parameter 1: The complete path to the LibZip source directory 
@echo                i.e: C:\local\src\libzip-1.1.3
@echo   Parameter 1: The complete path to the ZLib source directory 
@echo                i.e: C:\local\src\zlib-1.2.8
@echo   Parameter 2: The target path for the created binaries
@echo                i.e. C:\local\zlib-vs15
@echo The sript will also set the environment variable ZLIB_PATH to 
@echo the target directory
@echo.
goto FIN

:FIN

REM vim:set expandtab shiftwidth=2 tabstop=2:
