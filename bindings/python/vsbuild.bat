@echo off
setlocal enabledelayedexpansion

REM Copyright (c) Microsoft. All rights reserved.
REM
REM Licensed under the MIT license. See LICENSE.md file in the project root
REM for full license information.
REM ==============================================================================

REM Grab the parameters
REM
REM Note: don't rely on environment variables, since properties may have been
REM overridden at msbuild invocation.

set p_OutDir=%~1
set p_DebugBuild=%~2
set p_GpuBuild=%~3
set p_CNTK_COMPONENT_VERSION=%~4
set p_SWIG_PATH=%~5
set p_CNTK_PY_VERSIONS=%~6
set p_CNTK_PY27_PATH=%~7
set p_CNTK_PY34_PATH=%~8
set p_CNTK_PY35_PATH=%~9

REM Construct p_CNTK_PY_VERSIONS if not explicitly defined
REM (Note: to disable Python build completely, no CNTK_PYx_PATH variable must be defined)
if not defined p_CNTK_PY_VERSIONS (
  REM Note: leading space doesn't hurt
  if defined p_CNTK_PY27_PATH set p_CNTK_PY_VERSIONS=!p_CNTK_PY_VERSIONS! 27
  if defined p_CNTK_PY34_PATH set p_CNTK_PY_VERSIONS=!p_CNTK_PY_VERSIONS! 34
  if defined p_CNTK_PY35_PATH set p_CNTK_PY_VERSIONS=!p_CNTK_PY_VERSIONS! 35
)

REM Validate p_CNTK_PY_VERSIONS contents.
for %%p in (%p_CNTK_PY_VERSIONS%) do (
  if not "%%~p" == "27" if not "%%~p" == "34" if not "%%~p" == "35" echo Build for unsupported Python version '%%~p' requested, stopping&exit /b 1
)

REM Validate p_CNTK_PY_VERSIONS contents.
REM (Note: Don't merge with above loop; more robust parsing)
set nothingToBuild=1
for %%p in (%p_CNTK_PY_VERSIONS%) do (
  call set extraPath=!p_CNTK_PY%%~p_PATH!
  if not defined extraPath echo Build for Python version '%%~p' requested, but CNTK_PY%%~p_PATH not defined, stopping&exit /b 1
  set nothingToBuild=
)
if defined nothingToBuild echo Python support not configured to build.&exit /b 0

if "%p_DebugBuild%" == "true" echo Currently no Python build for Debug configurations, exiting.&exit /b 0

call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall" amd64
set CNTK_LIB_PATH=%p_OutDir%
set DIST_DIR=%p_OutDir%\Python
set PATH=%p_SWIG_PATH%;%PATH%
set CNTK_COMPONENT_VERSION=%p_CNTK_COMPONENT_VERSION%
set MSSdk=1
set DISTUTILS_USE_SDK=1

REM Build everything in supplied order
set oldPath=%PATH%
for %%p in (%p_CNTK_PY_VERSIONS%) do (
  call set extraPath=!p_CNTK_PY%%~p_PATH!
  echo Building for Python version '%%~p', extra path is !extraPath!
  set PATH=!extraPath!;!oldPath!
  python.exe .\setup.py ^
      build_ext --inplace --force --compiler msvc --plat-name=win-amd64 ^
      bdist_wheel --dist-dir "%DIST_DIR%"
  if errorlevel 1 exit /b 1
)
