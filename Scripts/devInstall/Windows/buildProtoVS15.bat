@REM Copyright (c) Microsoft. All rights reserved.
@REM Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
@REM
@REM batch script to build protobuf library for CNTK

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
if not "%3"=="" ( goto HELP )

SET SOURCEDIR=%~f1
SET TARGETDIR=%~f2

IF "%SOURCEDIR:~-1%"=="\" SET SOURCEDIR=%SOURCEDIR:~,-1%
IF "%TARGETDIR:~-1%"=="\" SET TARGETDIR=%TARGETDIR:~,-1%

if not exist "%SOURCEDIR%\build" (
  @echo Error: "%SOURCEDIR%" is not a valid ProtoBuf source directory
  goto FIN
)

set cmakePath=
for %%f in ("cmake.exe") do set cmakePath=%%~$PATH:f
if "%cmakePath%"=="" (
  @echo Error: CMAKE.EXE not found in PATH!
  goto FIN
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

@echo.
@echo This will build Protobuf-3.1.0 using Visual Studio 2015
@echo -------------------------------------------------------
@echo The configured settings for the batch file:
@echo    Visual Studio directory: %VCDIRECTORY%
@echo    Protobuf source directory: %SOURCEDIR%
@echo    Protobuf target directory: %TARGETDIR%
@echo.

pause 

call "%VCDIRECTORY%\..\..\VC\vcvarsall.bat" amd64 

pushd "%SOURCEDIR%"
cd cmake
md build && cd build

md debug && cd debug
cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Debug -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -DCMAKE_INSTALL_PREFIX="%TARGETDIR%" ..\..
nmake 
nmake install
cd ..

md release && cd release
cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -DCMAKE_INSTALL_PREFIX="%TARGETDIR%" ..\..
nmake 
nmake install
cd ..

popd

setx PROTOBUF_PATH %TARGETDIR%
set PROTOBUF_PATH=%TARGETDIR%

goto FIN



:HELP
@echo.
@echo Use this script to build the Protobuf library for CNTK.
@echo The script requires two parameter
@echo   Parameter 1: The complete path to the ProtoBuf source directory 
@echo                i.e: C:\local\src\protobuf-3.1.0
@echo   Parameter 2: The target path for the created binaries
@echo                i.e. C:\local\protobuf-3.1.0-vs15
@echo The sript will also set the environment variable PROTOBUF_PATH to 
@echo the target directory
@echo.
goto FIN

:FIN


REM vim:set expandtab shiftwidth=2 tabstop=2:
