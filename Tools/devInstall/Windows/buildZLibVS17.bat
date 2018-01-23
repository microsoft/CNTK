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

if not defined VS2017INSTALLDIR (
  @echo Environment variable VS2017INSTALLDIR not defined.
  @echo Make sure Visual Studion 2017 is installed.
  goto FIN
)
set VCDIRECTORY=%VS2017INSTALLDIR%
if "%VCDIRECTORY:~-1%"=="\" set VCDIRECTORY=%VCDIRECTORY:~,-1%

if not exist "%VCDIRECTORY%\VC\Auxiliary\Build\vcvarsall.bat" (
  echo Error: "%VCDIRECTORY%\VC\Auxiliary\Build\vcvarsall.bat" not found.
  echo Make sure you have installed Visual Studion 2017 correctly.
  goto FIN
)

if not exist "%VCDIRECTORY%\VC\Auxiliary\Build\14.11\Microsoft.VCToolsVersion.14.11.props" (
  echo Error: "%VCDIRECTORY%\VC\Auxiliary\Build\14.11\Microsoft.VCToolsVersion.14.11.props" not found.
  echo Make sure you have installed VCTools 14.11 for Visual Studio 2017 correctly.
  goto FIN
)

@set CMAKEGEN="Visual Studio 15 Win64"

@echo.
@echo This will build cthe compression library used by the CNTK image reader
@echo ----------------------------------------------------------------------
@echo The configured settings for the batch file:
@echo    Visual Studio directory: %VCDIRECTORY%
@echo    CMake Generator: %CMAKEGEN%
@echo    LibZip source directory: %LIBZIPSOURCEDIR%
@echo    Zlib source directory: %ZLIBSOURCEDIR%
@echo    Zlib-VS17 target directory: %TARGETDIR%
@echo.

@pause 

call "%VCDIRECTORY%\VC\Auxiliary\Build\vcvarsall.bat" amd64 --vcvars_ver=14.11

@pushd "%ZLIBSOURCEDIR%"
if exist build (rd /s /q build)
@mkdir build
@cd build
@cmake .. -G%CMAKEGEN% -DCMAKE_INSTALL_PREFIX="%TARGETDIR%"
@msbuild /P:Configuration=Release INSTALL.vcxproj
@popd

@pushd "%LIBZIPSOURCEDIR%"
if exist build (rd /s /q build)
@md build
@cd build
@cmake .. -G%CMAKEGEN% -DCMAKE_INSTALL_PREFIX="%TARGETDIR%"
@msbuild libzip.sln /t:zip /P:Configuration=Release
@cmake -DBUILD_TYPE=Release -P cmake_install.cmake
@popd

goto FIN

:HELP
@echo.
@echo Use this script to build the compression library used by the CNTK image reader
@echo The script requires three parameter
@echo   Parameter 1: The complete path to the LibZip source directory 
@echo                e.g C:\local\src\libzip-1.1.3
@echo   Parameter 2: The complete path to the ZLib source directory 
@echo                e.g C:\local\src\zlib-1.2.8
@echo   Parameter 3: The target path for the created binaries
@echo                e.g C:\local\zlib-vs17
@echo.
goto FIN

:FIN
endlocal

REM vim:set expandtab shiftwidth=2 tabstop=2:
