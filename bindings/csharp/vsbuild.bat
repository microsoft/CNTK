REM @echo off
setlocal enabledelayedexpansion

REM Copyright (c) Microsoft. All rights reserved.
REM
REM Licensed under the MIT license. See LICENSE.md file in the project root
REM for full license information.
REM ==============================================================================

set PROJECT_NAME=%~1
set SWIG_FILE="%~2"
set INC_DIR="%~3"
set OUTPUT_DIR="%~3"

echo Building %PROJECT_NAME% using swig

if not defined SWIG_PATH (
  echo SWIG_PATH not defined. 
  echo WARNING: the project %PROJECT_NAME% is not configured to build. 
  exit /b 1
)

if not exist %SWIG_PATH%\swig.exe (
  echo Cannot find swig.exe in %SWIG_PATH%
  echo WARNING: the project %PROJECT_NAME% is not configured to build. 
  exit /b 1
)

if not exist %SWIG_FILE% (
  echo Cannot find the SWIG interface file to build.
  exit /b 1
)

if not exist %OUTPUT_DIR% (
  mkdir %OUTPUT_DIR%
)

set SWIG_FLAGS=-c++ -csharp -DMSC_VER -I%INC_DIR% -outdir %OUTPUT_DIR%

echo Swigging %SWIG_FILE% ...
set CMD=%SWIG_PATH%\swig.exe %SWIG_FLAGS% %SWIG_FILE%
echo %CMD%
%CMD%


